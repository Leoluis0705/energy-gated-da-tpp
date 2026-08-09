"""Build the supplementary parameter-calibration table from archived evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selection_seeds(row: pd.Series) -> str:
    seeds = row.get("seeds")
    if pd.notna(seeds) and str(seeds).strip():
        return ";".join(str(int(float(item))) for item in str(seeds).split(";"))
    seed = row.get("seed")
    if pd.notna(seed):
        return str(int(float(seed)))
    raise ValueError(f"row has no auditable seed evidence: {row.get('stage')}")


def _contains_any_seed(serialized: str, prohibited: set[int]) -> bool:
    return any(int(item) in prohibited for item in serialized.split(";") if item)


def build_calibration_table(
    search_history_path: Path,
    frozen_protocol_path: Path,
    *,
    freeze_commit: str,
    freeze_time: str,
) -> pd.DataFrame:
    history = pd.read_csv(search_history_path)
    protocol = json.loads(frozen_protocol_path.read_text(encoding="utf-8"))

    table = history.copy()
    table["selection_data_seeds"] = table.apply(_selection_seeds, axis=1)
    formal_seeds = set(int(seed) for seed in protocol["allowed_seeds"])
    table["used_formal_evaluation_seeds"] = table["selection_data_seeds"].map(
        lambda value: _contains_any_seed(value, formal_seeds)
    )
    if table["used_formal_evaluation_seeds"].any():
        raise ValueError("formal evaluation seeds appear in calibration evidence")

    selected = (
        table["stage"].eq("weight_seeds0_4")
        & table["selection_rank"].eq(1)
        & table["M0"].eq(protocol["M0"])
        & table["G0"].eq(protocol["G0"])
        & table["alpha"].eq(protocol["alpha"])
        & table["beta"].eq(protocol["beta"])
        & table["gamma"].eq(protocol["gamma"])
        & table["mc_passes"].eq(protocol["mc_passes"])
    )
    if int(selected.sum()) != 1:
        raise ValueError("search history does not identify exactly one frozen final choice")
    table["selected_final_protocol"] = selected

    table["selection_primary_metric"] = "mean_AUTC"
    table["selection_tie_tolerance"] = 1e-4
    table["selection_tie_rule"] = (
        "fewer correction rounds; then smaller fixed grid distance from original center"
    )
    table["cohort_role"] = "development_only"
    table["search_history_path"] = search_history_path.as_posix()
    table["search_history_sha256"] = sha256_file(search_history_path)
    table["frozen_protocol_path"] = frozen_protocol_path.as_posix()
    table["frozen_protocol_sha256"] = sha256_file(frozen_protocol_path)
    table["freeze_git_commit"] = freeze_commit
    table["freeze_time"] = freeze_time
    table["freeze_time_evidence"] = "git_commit_time"
    return table


def _write_without_overwriting_different_content(table: pd.DataFrame, output: Path) -> None:
    rendered = table.to_csv(index=False, lineterminator="\n")
    if output.exists():
        if output.read_text(encoding="utf-8") != rendered:
            raise FileExistsError(f"refusing to overwrite non-identical evidence: {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8", newline="")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--search-history",
        type=Path,
        default=Path("results/parameter_selection/search_history.csv"),
    )
    parser.add_argument(
        "--protocol", type=Path, default=Path("configs/frozen_final_protocol.yaml")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("supplementary/tables/table_s_parameter_calibration.csv"),
    )
    parser.add_argument(
        "--freeze-commit",
        default="8a7599807ac786c9bb664bb199dc5bf422db6f93",
    )
    parser.add_argument("--freeze-time", default="2026-07-18T03:19:53+08:00")
    args = parser.parse_args()

    table = build_calibration_table(
        args.search_history,
        args.protocol,
        freeze_commit=args.freeze_commit,
        freeze_time=args.freeze_time,
    )
    _write_without_overwriting_different_content(table, args.output)
    print(f"validated {len(table)} calibration rows -> {args.output}")


if __name__ == "__main__":
    main()
