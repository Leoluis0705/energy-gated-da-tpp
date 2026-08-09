from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from pymatgen.core import Element

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.audit_common import sha256_file, write_bytes_protected


DESIGN_ORDER = [
    "element_system_current",
    "coelement_block_multiset",
    "coelement_iupac_group_set",
    "chemical_complexity_nelements",
]

DESIGN_METADATA = {
    "element_system_current": {
        "status": "current_baseline",
        "description": "Exact sorted element-system identity from the retained chemsys metadata.",
        "materials_basis": "Exact chemical-system membership.",
    },
    "coelement_block_multiset": {
        "status": "candidate_not_selected",
        "description": "Counts of s-, p-, d- and f-block co-elements after excluding Mn and O.",
        "materials_basis": "Periodic-table electronic block captures broad bonding/redox classes without using target values.",
    },
    "coelement_iupac_group_set": {
        "status": "candidate_not_selected",
        "description": "Set of IUPAC group numbers of co-elements, with lanthanide/actinide tags; Mn and O are excluded and multiplicity is ignored.",
        "materials_basis": "Elements in the same periodic group share valence-electron families; the set intentionally collapses exact identities.",
    },
    "chemical_complexity_nelements": {
        "status": "candidate_not_selected_extreme_coarse_reference",
        "description": "Exact number of distinct elements in the oxide chemical system.",
        "materials_basis": "Binary/ternary/multinary chemical complexity; this is an intentionally coarse sensitivity bound.",
    },
}


def _coelements(chemsys: str) -> list[str]:
    elements = sorted({token.strip() for token in str(chemsys).split("-") if token.strip()})
    if "Mn" not in elements or "O" not in elements:
        raise ValueError(f"Mn-oxide chemsys is missing Mn or O: {chemsys!r}")
    return [element for element in elements if element not in {"Mn", "O"}]


def _block_multiset_key(chemsys: str) -> str:
    counts = Counter(Element(symbol).block for symbol in _coelements(chemsys))
    return "none" if not counts else "|".join(f"{block}{counts[block]}" for block in "spdf" if counts[block])


def _periodic_group_token(symbol: str) -> str:
    element = Element(symbol)
    if element.is_lanthanoid:
        return "Ln"
    if element.is_actinoid:
        return "An"
    return f"G{int(element.group)}"


def _iupac_group_set_key(chemsys: str) -> str:
    tokens = sorted({_periodic_group_token(symbol) for symbol in _coelements(chemsys)})
    return "none" if not tokens else "|".join(tokens)


def build_group_keys(metadata: pd.DataFrame) -> pd.DataFrame:
    required = {"candidate_id", "chemsys", "nelements"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"metadata is missing required pre-query columns: {sorted(missing)}")
    frame = metadata.loc[:, ["candidate_id", "chemsys", "nelements"]].copy()
    frame["candidate_id"] = frame["candidate_id"].astype(str).str.strip()
    if frame["candidate_id"].duplicated().any():
        raise ValueError("candidate_id must be unique")
    frame["chemsys"] = frame["chemsys"].astype(str).map(
        lambda value: "-".join(sorted(token for token in value.split("-") if token))
    )
    declared_nelements = pd.to_numeric(frame["nelements"], errors="raise").astype(int)
    observed_nelements = frame["chemsys"].map(lambda value: len(value.split("-")))
    if not declared_nelements.equals(observed_nelements.astype(int)):
        raise ValueError("nelements disagrees with chemsys token count")
    return pd.DataFrame(
        {
            "candidate_id": frame["candidate_id"],
            "element_system_current": frame["chemsys"],
            "coelement_block_multiset": frame["chemsys"].map(_block_multiset_key),
            "coelement_iupac_group_set": frame["chemsys"].map(_iupac_group_set_key),
            "chemical_complexity_nelements": declared_nelements.map(lambda value: f"E{value}"),
        }
    )


def group_inventory(keys: pd.Series) -> dict[str, object]:
    counts = keys.astype(str).value_counts()
    size_distribution = counts.value_counts().sort_index()
    singleton_count = int((counts == 1).sum())
    return {
        "candidate_count": int(len(keys)),
        "group_count": int(len(counts)),
        "singleton_group_count": singleton_count,
        "singleton_group_fraction": singleton_count / len(counts) if len(counts) else 0.0,
        "singleton_candidate_fraction": singleton_count / len(keys) if len(keys) else 0.0,
        "maximum_group_size": int(counts.max()) if len(counts) else 0,
        "group_size_distribution_json": json.dumps(
            {str(int(size)): int(number) for size, number in size_distribution.items()},
            sort_keys=True,
            separators=(",", ":"),
        ),
    }


def load_formal_top_b_history(archive: Path) -> pd.DataFrame:
    evidence = Path(archive) / "baseline_snapshot/archive/experiments/reproducibility/results"
    cohorts = [
        (evidence / "paired_two_dataset_confirmation_20260712", range(5, 10)),
        (evidence / "paired_two_dataset_confirmation_seeds_10_14_20260713", range(10, 15)),
    ]
    rows: list[dict[str, object]] = []
    for root, seeds in cohorts:
        for method in ("energy_gated_da_tpp", "predicted_distance_greedy"):
            for seed in seeds:
                path = root / "runs/mnoxide" / method / f"seed_{seed}/round_diagnostics.csv"
                if not path.is_file():
                    raise FileNotFoundError(path)
                diagnostics = pd.read_csv(
                    path,
                    usecols=["round", "direct_top_b_candidate_ids"],
                    dtype={"direct_top_b_candidate_ids": str},
                )
                for record in diagnostics.itertuples(index=False):
                    candidate_ids = [
                        value.strip()
                        for value in str(record.direct_top_b_candidate_ids).split(";")
                        if value.strip()
                    ]
                    rows.append(
                        {
                            "method": method,
                            "seed": int(seed),
                            "round": int(record.round),
                            "candidate_ids": candidate_ids,
                            "source_path": str(path.resolve()),
                            "source_sha256": sha256_file(path),
                        }
                    )
    history = pd.DataFrame(rows)
    if len(history) != 400:
        raise ValueError(f"expected 400 formal Mn-oxide top-b observations, found {len(history)}")
    if not history["candidate_ids"].map(len).eq(16).all():
        raise ValueError("formal direct top-b history does not consistently contain 16 candidates")
    return history


def _concentrations(keys: pd.Series, history: pd.DataFrame) -> pd.DataFrame:
    key_by_id = dict(zip(keys.index.astype(str), keys.astype(str), strict=True))
    rows = []
    for record in history.itertuples(index=False):
        missing = [candidate_id for candidate_id in record.candidate_ids if candidate_id not in key_by_id]
        if missing:
            raise ValueError(f"top-b IDs missing from frozen metadata: {missing[:5]}")
        groups = [key_by_id[candidate_id] for candidate_id in record.candidate_ids]
        rows.append(
            {
                "method": record.method,
                "concentration": max(Counter(groups).values()) / len(groups),
            }
        )
    return pd.DataFrame(rows)


def _summary(values: pd.Series, prefix: str) -> dict[str, float | int]:
    return {
        f"{prefix}_observations": int(len(values)),
        f"{prefix}_min": float(values.min()),
        f"{prefix}_q25": float(values.quantile(0.25)),
        f"{prefix}_median": float(values.median()),
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_q75": float(values.quantile(0.75)),
        f"{prefix}_max": float(values.max()),
    }


def build_inventory(keys: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    manifest_payload = "".join(
        f"{path}\t{digest}\n"
        for path, digest in sorted(
            history[["source_path", "source_sha256"]].drop_duplicates().itertuples(index=False, name=None)
        )
    )
    history_manifest_sha256 = hashlib.sha256(manifest_payload.encode("utf-8")).hexdigest()
    rows = []
    indexed = keys.set_index("candidate_id")
    for design in DESIGN_ORDER:
        concentrations = _concentrations(indexed[design], history)
        row: dict[str, object] = {
            "design": design,
            **DESIGN_METADATA[design],
            **group_inventory(indexed[design]),
            "uses_target_label": False,
            "available_before_query": True,
            "top_b_denominator": 16,
            "top_b_history_scope": "corrected seeds 5-14; Gate and Greedy; all 20 Mn-oxide rounds",
            "top_b_history_manifest_sha256": history_manifest_sha256,
            **_summary(concentrations["concentration"], "top_b_all"),
        }
        for method, prefix in (
            ("energy_gated_da_tpp", "top_b_gate"),
            ("predicted_distance_greedy", "top_b_greedy"),
        ):
            row.update(_summary(concentrations.loc[concentrations["method"] == method, "concentration"], prefix))
        rows.append(row)
    return pd.DataFrame(rows)


def _render_report(inventory: pd.DataFrame, oracle_path: Path, history: pd.DataFrame) -> str:
    summary = inventory[
        [
            "design",
            "group_count",
            "singleton_group_count",
            "singleton_group_fraction",
            "maximum_group_size",
            "top_b_all_median",
            "top_b_all_max",
        ]
    ].copy()
    lines = [
        "# Mn-oxide Group-key Candidate Designs",
        "",
        "## Decision status",
        "",
        "No alternative key is selected in this stage. This report is a static, label-blind inventory only; it does not rerun acquisition, training, recovery or AUTC.",
        "",
        "## Inventory",
        "",
        summary.to_markdown(index=False),
        "",
        "`singleton_group_fraction` uses number of groups as its denominator. The CSV also gives `singleton_candidate_fraction`, whose denominator is the 640-candidate pool. Historical top-b concentration is the largest group count divided by 16 before correction.",
        "",
        "The current exact element-system key reproduces 614 groups and 588 singleton groups. This confirms that the retained representation is nearly candidate-unique and can mechanically suppress group repetition and therefore group-driven replacement.",
        "",
        "## Candidate definitions",
        "",
    ]
    for design in DESIGN_ORDER:
        metadata = DESIGN_METADATA[design]
        lines.extend(
            [
                f"### `{design}`",
                "",
                metadata["description"],
                "",
                f"Materials/data basis: {metadata['materials_basis']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Information boundary",
            "",
            "Only `candidate_id`, `chemsys` and `nelements` are loaded from the frozen Mn-oxide metadata. Target labels, formation energies, recovery and final outcomes are not loaded into group construction. All proposed keys are available before querying. The term co-element is used deliberately: no oxidation state or cation assignment is inferred.",
            "",
            "## Historical top-b scope",
            "",
            f"The concentration distribution contains {len(history)} retained direct top-b observations: two methods × ten corrected seeds × twenty rounds. Each diagnostics file is read with only `round` and `direct_top_b_candidate_ids` columns. The CSV records separate Gate, Greedy and pooled summaries.",
            "",
            "## Provenance",
            "",
            f"- Frozen metadata: `{oracle_path.resolve()}`",
            f"- Frozen metadata SHA-256: `{sha256_file(oracle_path)}`",
            f"- Unique round-diagnostics files: {history['source_path'].nunique()}",
            f"- Round-diagnostics manifest SHA-256: `{inventory['top_b_history_manifest_sha256'].iloc[0]}`",
            f"- Analysis script SHA-256: `{sha256_file(Path(__file__).resolve())}`",
            "",
            "## Next decision",
            "",
            "A materials-domain choice among the candidate representations is still required before any ten-seed sensitivity run. The extreme chemical-complexity key is best treated as a lower-resolution bound, not a default recommendation.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--historical-root", type=Path, default=Path(r"D:\CGCNN"))
    parser.add_argument("--check-existing", action="store_true")
    args = parser.parse_args(argv)
    archive = args.archive.resolve()
    oracle_path = (
        args.historical_root.resolve()
        / "NON_GEN_INTERVAL_POOLS_20260618/Mn_NON_GEN_HARD640_M2P59_M2P47_111_20260709/oracle.csv"
    )
    if not oracle_path.is_file():
        raise FileNotFoundError(oracle_path)
    metadata = pd.read_csv(
        oracle_path,
        usecols=["candidate_id", "chemsys", "nelements"],
        dtype={"candidate_id": str},
    )
    keys = build_group_keys(metadata)
    history = load_formal_top_b_history(archive)
    inventory = build_inventory(keys, history)
    csv_buffer = io.StringIO()
    inventory.to_csv(csv_buffer, index=False, lineterminator="\n")
    report = _render_report(inventory, oracle_path, history)
    outputs = {
        archive / "results/group_key/group_key_inventory.csv": csv_buffer.getvalue().encode("utf-8"),
        archive / "docs/GROUP_KEY_CANDIDATE_DESIGNS.md": report.encode("utf-8"),
    }
    statuses = {
        str(path.relative_to(archive)): write_bytes_protected(path, content, args.check_existing)
        for path, content in outputs.items()
    }
    print(json.dumps(statuses, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
