"""Write the immutable protocol bundle after development-only parameter selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from experiments.reproducibility.formal_protocol import load_formal_protocol


PROTOCOL_VERSION = "egdatpp_psfix_v1"
FINAL_SEEDS = list(range(15, 25))
MC_SENSITIVITY_SEEDS = list(range(25, 30))
LIMO_METHODS = [
    "interval_hit_greedy",
    "always_da_tpp",
    "margin_only_gate",
    "group_only_gate",
    "energy_gated_da_tpp",
]
MC_METHODS = ["interval_hit_greedy", "energy_gated_da_tpp"]
GROUP_MAP_ROOT = "configs/group_keys/egdatpp_psfix_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_yaml(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with path.open("xb") as handle:
        handle.write(data)


def _protocol(
    *,
    phase: str,
    dataset: str,
    allowed_seeds: list[int],
    allowed_methods: list[str],
    mc_passes: int,
    parameters: dict[str, float],
    group_key_mode: str,
    group_key_map_relative_path: str | None,
) -> dict[str, object]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "phase": phase,
        "dataset": dataset,
        "allowed_seeds": allowed_seeds,
        "allowed_methods": allowed_methods,
        "mc_passes": mc_passes,
        **parameters,
        "group_key_mode": group_key_mode,
        "group_key_map_relative_path": group_key_map_relative_path,
        "frozen": True,
    }


def build_frozen_protocol_bundle(
    *,
    configs_root: Path,
    mc_passes: int,
    m0: float,
    g0: float,
    alpha: float,
    beta: float,
    gamma: float,
    git_commit: str,
    source_evidence: dict[str, str],
) -> dict[str, Path]:
    """Create all final-evaluation protocols without overwriting existing files."""

    if mc_passes not in {3, 10, 30}:
        raise ValueError("mc_passes must be one of 3, 10, 30")
    parameters = {
        "M0": float(m0),
        "G0": float(g0),
        "alpha": float(alpha),
        "beta": float(beta),
        "gamma": float(gamma),
    }
    if not all(math.isfinite(value) for value in parameters.values()):
        raise ValueError("all selected parameters must be finite")
    root = Path(configs_root)
    protocol_root = root / "frozen_protocols" / PROTOCOL_VERSION
    primary = root / "frozen_final_protocol.yaml"
    payloads: dict[Path, dict[str, object]] = {
        primary: _protocol(
            phase="formal_evaluation",
            dataset="limo",
            allowed_seeds=FINAL_SEEDS,
            allowed_methods=LIMO_METHODS,
            mc_passes=mc_passes,
            parameters=parameters,
            group_key_mode="element_system_current",
            group_key_map_relative_path=None,
        ),
        protocol_root / "mnoxide_original.yaml": _protocol(
            phase="formal_evaluation",
            dataset="mnoxide",
            allowed_seeds=FINAL_SEEDS,
            allowed_methods=[
                "interval_hit_greedy",
                "always_da_tpp",
                "energy_gated_da_tpp",
            ],
            mc_passes=mc_passes,
            parameters=parameters,
            group_key_mode="element_system_current",
            group_key_map_relative_path=(
                f"{GROUP_MAP_ROOT}/mnoxide_element_system_current.csv"
            ),
        ),
        protocol_root / "mnoxide_block.yaml": _protocol(
            phase="formal_evaluation",
            dataset="mnoxide",
            allowed_seeds=FINAL_SEEDS,
            allowed_methods=["energy_gated_da_tpp"],
            mc_passes=mc_passes,
            parameters=parameters,
            group_key_mode="coelement_block_multiset",
            group_key_map_relative_path=(
                f"{GROUP_MAP_ROOT}/mnoxide_coelement_block_multiset.csv"
            ),
        ),
        protocol_root / "mnoxide_iupac.yaml": _protocol(
            phase="formal_evaluation",
            dataset="mnoxide",
            allowed_seeds=FINAL_SEEDS,
            allowed_methods=["energy_gated_da_tpp"],
            mc_passes=mc_passes,
            parameters=parameters,
            group_key_mode="coelement_iupac_group_set",
            group_key_map_relative_path=(
                f"{GROUP_MAP_ROOT}/mnoxide_coelement_iupac_group_set.csv"
            ),
        ),
    }
    for k in (3, 10, 30):
        payloads[protocol_root / f"limo_mc_k{k}.yaml"] = _protocol(
            phase="mc_dropout_sensitivity",
            dataset="limo",
            allowed_seeds=MC_SENSITIVITY_SEEDS,
            allowed_methods=MC_METHODS,
            mc_passes=k,
            parameters=parameters,
            group_key_mode="element_system_current",
            group_key_map_relative_path=None,
        )

    for path, payload in payloads.items():
        _write_json_yaml(path, payload)
        load_formal_protocol(path)

    manifest = protocol_root / "frozen_protocol_manifest.json"
    manifest_payload: dict[str, object] = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": str(git_commit),
        "selected_parameters": {"mc_passes": mc_passes, **parameters},
        "source_evidence": dict(sorted(source_evidence.items())),
        "protocols": [
            {
                "relative_path": path.relative_to(root).as_posix(),
                "sha256": _sha256(path),
            }
            for path in sorted(payloads, key=lambda item: item.as_posix())
        ],
    }
    _write_json_yaml(manifest, manifest_payload)
    return {"primary": primary, "manifest": manifest, **{path.name: path for path in payloads}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs-root", type=Path, required=True)
    parser.add_argument("--mc-passes", type=int, required=True)
    parser.add_argument("--m0", type=float, required=True)
    parser.add_argument("--g0", type=float, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--source-evidence-json", type=Path, required=True)
    args = parser.parse_args()
    source_evidence = json.loads(args.source_evidence_json.read_text(encoding="utf-8"))
    outputs = build_frozen_protocol_bundle(
        configs_root=args.configs_root,
        mc_passes=args.mc_passes,
        m0=args.m0,
        g0=args.g0,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        git_commit=args.git_commit,
        source_evidence=source_evidence,
    )
    print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
