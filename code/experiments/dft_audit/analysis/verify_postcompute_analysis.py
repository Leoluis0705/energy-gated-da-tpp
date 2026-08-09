"""Verify a generated post-compute analysis bundle without changing it."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.audit_common import sha256_file
from analysis.postprocess_formal_results import (
    formation_energy_per_atom,
    validate_formal_gpu_grid,
)


RESTRICTED_VASP_NAMES = re.compile(
    r"^(POTCAR|WAVECAR|CHGCAR|CHG|AECCAR0|AECCAR1|AECCAR2)(\.|$)", re.IGNORECASE
)
PROHIBITED_REPORT_PHRASES = (
    "ground state",
    "ground-state",
    "exhaustive magnetic search",
    "no abnormal short bonds",
    "no cell collapse",
)


def verify_bundle(project_root: Path, output_root: Path) -> dict[str, object]:
    project_root = Path(project_root).resolve()
    output_root = Path(output_root).resolve()
    if not output_root.is_dir():
        raise FileNotFoundError(output_root)

    inventory = pd.read_csv(output_root / "SHA256SUMS.csv")
    actual_paths = sorted(
        path.relative_to(output_root).as_posix()
        for path in output_root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS.csv"
    )
    expected_paths = sorted(inventory["relative_path"].astype(str))
    if actual_paths != expected_paths:
        raise ValueError("generated-file inventory does not match the bundle")
    for row in inventory.to_dict(orient="records"):
        path = output_root / Path(str(row["relative_path"]))
        if path.stat().st_size != int(row["size_bytes"]):
            raise ValueError(f"size mismatch: {path}")
        if sha256_file(path) != str(row["sha256"]):
            raise ValueError(f"SHA-256 mismatch: {path}")

    restricted = [
        path for path in output_root.rglob("*") if path.is_file() and RESTRICTED_VASP_NAMES.match(path.name)
    ]
    if restricted:
        raise ValueError(f"restricted VASP files found: {restricted}")

    metrics = pd.read_csv(output_root / "gpu" / "per_seed_metrics.csv")
    validate_formal_gpu_grid(metrics)
    if len(metrics) != 130:
        raise ValueError("formal GPU metric table is not 130 rows")
    if metrics["reported_minus_recomputed_AUTC"].abs().max() > 1e-12:
        raise ValueError("reported GPU AUTC differs from raw-history recomputation")
    if not (metrics["formal_protocol_sha256"] == metrics["config_hash"]).all():
        raise ValueError("a task protocol hash differs from its manifest config hash")
    expected_gpu_protocol = "2a8d0fb5114c0e2f2457d9887ff5cfc6b8c9ff701669f473e018984855fcac84"
    li_formal = metrics[metrics["formal_stage"] == "li_m_o_ablation"]
    if set(li_formal["formal_protocol_sha256"]) != {expected_gpu_protocol}:
        raise ValueError("unexpected frozen Li-M-O formal protocol hash")
    if metrics["formal_protocol_sha256"].nunique() != 7:
        raise ValueError("unexpected number of frozen/derived task protocol hashes")

    masks = pd.read_csv(output_root / "gpu" / "dropout_mask_pairing_audit.csv")
    if not masks["all_paired_mask_sequences_identical"].all():
        raise ValueError("paired method mask sequence differs")
    paired = json.loads((output_root / "gpu" / "paired_statistics.json").read_text(encoding="utf-8"))
    environment = paired["environment"]
    if environment["bootstrap_samples"] != 100_000 or environment["bootstrap_seed"] != 20260719:
        raise ValueError("unexpected bootstrap settings")
    expected_wilcoxon = {
        "zero_method": "wilcox",
        "correction": False,
        "alternative": "two-sided",
        "method": "exact",
    }
    if environment["wilcoxon"] != expected_wilcoxon:
        raise ValueError("unexpected Wilcoxon settings")

    settings = pd.read_csv(output_root / "dft" / "dft_settings.csv")
    convergence = pd.read_csv(output_root / "dft" / "convergence_inventory.csv")
    structures = pd.read_csv(output_root / "dft" / "structure_metrics.csv")
    formation = pd.read_csv(output_root / "dft" / "recomputed_formation_energies.csv")
    references = pd.read_csv(output_root / "dft" / "elemental_references.csv")
    manifest = pd.read_csv(output_root / "dft" / "dft_candidate_manifest.csv")
    if (len(settings), len(convergence), len(structures), len(formation), len(manifest)) != (21, 21, 21, 18, 20):
        raise ValueError("unexpected DFT output row counts")
    if not convergence["electronic_converged"].all():
        raise ValueError("a frozen-protocol static record is not electronically converged")
    if manifest["pilot_or_new"].value_counts().to_dict() != {"new": 12, "pilot": 8}:
        raise ValueError("candidate cohort counts differ from 12 new plus 8 pilot")
    if int(manifest["main_text_selected"].sum()) != 3:
        raise ValueError("main-text candidate count is not three")

    for row in formation.to_dict(orient="records"):
        ref_subset = references[references["functional"] == row["functional"]]
        ref_map = dict(zip(ref_subset["element"], ref_subset["energy_per_atom_eV"], strict=True))
        recomputed = formation_energy_per_atom(
            row["final_total_energy_eV"],
            Composition(row["formula"]).as_dict(),
            ref_map,
        )
        if not np.isclose(recomputed, row["formation_energy_eV_per_atom"], atol=1e-12, rtol=0):
            raise ValueError(f"formation-energy formula mismatch: {row['candidate_id']}")
    expected_dft_protocol = "4544f2a9a4685399ee4aaa34de4368f17febe846cca9b1fb8564e9a8306e5b7e"
    if set(settings["frozen_protocol_sha256"]) != {expected_dft_protocol}:
        raise ValueError("unexpected DFT frozen-protocol hash")

    for path in (output_root / "reports").glob("*.md"):
        text = path.read_text(encoding="utf-8").lower()
        matched = [phrase for phrase in PROHIBITED_REPORT_PHRASES if phrase in text]
        if matched:
            raise ValueError(f"prohibited report phrase in {path}: {matched}")

    for path in (output_root / "figures").glob("*.pdf"):
        if path.read_bytes()[:4] != b"%PDF":
            raise ValueError(f"invalid PDF signature: {path}")
    png_signature = b"\x89PNG\r\n\x1a\n"
    for path in (output_root / "figures").glob("*.png"):
        if path.read_bytes()[:8] != png_signature:
            raise ValueError(f"invalid PNG signature: {path}")

    v33_reference = json.loads((project_root / "analysis" / "v33_table_reference.json").read_text(encoding="utf-8"))
    v33_path = Path(v33_reference["source"]["path"])
    if not v33_path.is_file() or sha256_file(v33_path) != v33_reference["source"]["sha256"]:
        raise ValueError("frozen v33 reference PDF is missing or changed")

    return {
        "verified_generated_files": len(inventory),
        "gpu_trajectories": len(metrics),
        "paired_mask_configurations": len(masks),
        "dft_static_records": len(settings),
        "formation_energy_records": len(formation),
        "candidate_manifest_records": len(manifest),
        "restricted_vasp_files": 0,
        "v33_sha256_unchanged": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(verify_bundle(args.project_root, args.output_root), indent=2, sort_keys=True))
