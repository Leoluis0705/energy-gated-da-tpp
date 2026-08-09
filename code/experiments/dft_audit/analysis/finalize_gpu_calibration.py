"""Finalize a completed GPU calibration stage into an immutable audit bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.analyze_gpu_parameter_calibration import (
    AUTC_TIE_TOLERANCE,
    aggregate_calibration_configs,
    build_combined_calibration_manifest,
    rank_aggregated_configs,
    summarize_completed_calibration,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_record(row: pd.Series) -> dict[str, object]:
    return json.loads(row.to_json())


def finalize_calibration_stage(
    *,
    screen_manifest: Path,
    promotion_manifest: Path,
    seed0_ranking: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Combine the selected seed-0 runs with seeds 1--4 and rank their means."""

    started_at = datetime.now(timezone.utc)
    output_dir.mkdir(parents=True, exist_ok=False)
    ranked_seed0 = pd.read_csv(seed0_ranking)
    combined_path = output_dir / "combined_manifest.csv"
    build_combined_calibration_manifest(
        screen_manifest_path=screen_manifest,
        promotion_manifest_path=promotion_manifest,
        ranked_seed0=ranked_seed0,
        manifest_path=combined_path,
    )
    per_seed = summarize_completed_calibration(combined_path)
    ranking = rank_aggregated_configs(
        aggregate_calibration_configs(per_seed, expected_seeds=range(5))
    )
    per_seed_path = output_dir / "per_seed_results.csv"
    ranking_path = output_dir / "full_ranking.csv"
    per_seed.to_csv(per_seed_path, index=False, lineterminator="\n")
    ranking.to_csv(ranking_path, index=False, lineterminator="\n")

    selected = _json_record(ranking.iloc[0])
    result: dict[str, object] = {
        "started_at": started_at.isoformat(),
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "selected_config_id": selected["config_id"],
        "selected_configuration": selected,
        "selection_policy": {
            "primary": "highest mean AUTC",
            "autc_tie_tolerance": AUTC_TIE_TOLERANCE,
            "secondary": "fewest mean correction rounds",
            "tertiary": "smallest distance from original center",
        },
        "development_seeds": [0, 1, 2, 3, 4],
        "source_sha256": {
            "screen_manifest": _sha256(screen_manifest),
            "promotion_manifest": _sha256(promotion_manifest),
            "seed0_ranking": _sha256(seed0_ranking),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "platform": platform.platform(),
        },
        "command": [sys.executable, *sys.argv],
    }
    selection_path = output_dir / "selection.json"
    selection_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    evidence = (combined_path, per_seed_path, ranking_path, selection_path)
    (output_dir / "SHA256SUMS").write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in evidence),
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen-manifest", type=Path, required=True)
    parser.add_argument("--promotion-manifest", type=Path, required=True)
    parser.add_argument("--seed0-ranking", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = finalize_calibration_stage(
        screen_manifest=args.screen_manifest,
        promotion_manifest=args.promotion_manifest,
        seed0_ranking=args.seed0_ranking,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
