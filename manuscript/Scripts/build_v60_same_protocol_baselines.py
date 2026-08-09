"""Summarize the source-backed same-protocol development baselines for V60."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


SEEDS = tuple(range(5))
DISPLAY = {
    "energy_gated_da_tpp": "Energy-Gated DA-TPP",
    "predicted_target_greedy": "Predicted-Target Greedy",
    "mc_dropout": "MC Dropout",
    "random_sampling": "Random Sampling",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_method(method: str, root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for seed in SEEDS:
        run = root / f"seed_{seed}" / "attempt_1"
        paths = {name: run / name for name in ("status.json", "run_config.json", "run_metrics.csv", "summary.csv")}
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError("; ".join(missing))
        status = json.loads(paths["status.json"].read_text(encoding="utf-8"))
        if status.get("status") != "DONE":
            raise RuntimeError(f"run is not DONE: {run}")
        metric_frame = pd.read_csv(paths["run_metrics.csv"])
        if len(metric_frame) != 1:
            raise ValueError(f"expected one metric row: {paths['run_metrics.csv']}")
        metric = metric_frame.iloc[0]
        trajectory = pd.read_csv(paths["summary.csv"])
        early = trajectory[trajectory["oracle_evaluations"] <= 160]
        if early.empty or int(early.iloc[-1]["oracle_evaluations"]) != 160:
            raise ValueError(f"missing 160-query row: {paths['summary.csv']}")
        sequence_hash = str(metric.get("candidate_sequence_sha256", ""))
        if not sequence_hash:
            selected = ";".join(trajectory["selected_candidate_ids"].astype(str))
            sequence_hash = hashlib.sha256(selected.encode("utf-8")).hexdigest()
        rows.append(
            {
                "method": method,
                "display_name": DISPLAY[method],
                "seed": seed,
                "AUTC_160": float(early.iloc[-1]["AUTC_so_far"]),
                "AUTC_640": float(metric["AUTC"]),
                "recovery_at_80": int(metric["recovery_at_80"]),
                "recovery_at_160": int(metric["recovery_at_160"]),
                "recovery_at_320": int(metric["recovery_at_320"]),
                "final_recovery": int(metric["final_recovery"]),
                "sequence_sha256": sequence_hash,
                "run_config_sha256": sha256(paths["run_config.json"]),
                "run_metrics_sha256": sha256(paths["run_metrics.csv"]),
                "summary_sha256": sha256(paths["summary.csv"]),
                "run_path": str(run),
            }
        )
    return pd.DataFrame(rows)


def aggregate(per_seed: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for method, block in per_seed.groupby("method", sort=False):
        record: dict[str, object] = {
            "method": method,
            "display_name": block.iloc[0]["display_name"],
            "seed_count": len(block),
            "unique_sequences": block["sequence_sha256"].nunique(),
        }
        for column in ("AUTC_160", "AUTC_640", "recovery_at_80", "recovery_at_160", "recovery_at_320", "final_recovery"):
            record[f"mean_{column}"] = block[column].mean()
            record[f"sd_{column}"] = block[column].std(ddof=1)
        records.append(record)
    return pd.DataFrame(records)


def paired_gate_greedy(per_seed: pd.DataFrame) -> pd.DataFrame:
    gate = per_seed[per_seed["method"] == "energy_gated_da_tpp"].copy()
    greedy = per_seed[per_seed["method"] == "predicted_target_greedy"].copy()
    paired = gate.merge(
        greedy,
        on="seed",
        suffixes=("_gate", "_greedy"),
        validate="one_to_one",
    )
    for metric in ("AUTC_160", "AUTC_640", "recovery_at_80", "recovery_at_160"):
        paired[f"delta_{metric}_gate_minus_greedy"] = (
            paired[f"{metric}_gate"] - paired[f"{metric}_greedy"]
        )
    delta = paired["delta_AUTC_160_gate_minus_greedy"]
    paired["AUTC_160_leader"] = np.select(
        [delta > 0, delta < 0],
        ["Gate", "Greedy"],
        default="Tie",
    )
    keep = [
        "seed",
        "AUTC_160_gate",
        "AUTC_160_greedy",
        "delta_AUTC_160_gate_minus_greedy",
        "AUTC_160_leader",
        "AUTC_640_gate",
        "AUTC_640_greedy",
        "delta_AUTC_640_gate_minus_greedy",
        "recovery_at_80_gate",
        "recovery_at_80_greedy",
        "delta_recovery_at_80_gate_minus_greedy",
        "recovery_at_160_gate",
        "recovery_at_160_greedy",
        "delta_recovery_at_160_gate_minus_greedy",
    ]
    return paired[keep].sort_values("seed").reset_index(drop=True)


def write_table(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Method & AUTC$_{160}$ & AUTC$_{640}$ & $R_{80}$ & $R_{160}$ & Final recovery & Unique paths \\",
        r"\midrule",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"{row.display_name} & ${row.mean_AUTC_160:.4f} \\pm {row.sd_AUTC_160:.4f}$ & "
            f"${row.mean_AUTC_640:.4f} \\pm {row.sd_AUTC_640:.4f}$ & "
            f"${row.mean_recovery_at_80:.1f} \\pm {row.sd_recovery_at_80:.1f}$ & "
            f"${row.mean_recovery_at_160:.1f} \\pm {row.sd_recovery_at_160:.1f}$ & "
            f"${row.mean_final_recovery:.1f} \\pm {row.sd_final_recovery:.1f}$ & "
            f"{int(row.unique_sequences)}/5 \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--si-table", type=Path, required=True)
    args = parser.parse_args()

    frames = [read_method("energy_gated_da_tpp", args.gate_root)]
    for method in ("predicted_target_greedy", "mc_dropout", "random_sampling"):
        frames.append(read_method(method, args.baseline_root / method))
    per_seed = pd.concat(frames, ignore_index=True)
    summary = aggregate(per_seed)
    paired = paired_gate_greedy(per_seed)
    args.source_dir.mkdir(parents=True, exist_ok=True)
    per_seed_path = args.source_dir / "v60_same_protocol_baselines_per_seed.csv"
    summary_path = args.source_dir / "v60_same_protocol_baselines_summary.csv"
    audit_path = args.source_dir / "v60_same_protocol_baselines_seed_audit.csv"
    paired_path = args.source_dir / "v60_same_protocol_gate_greedy_paired.csv"
    per_seed.to_csv(per_seed_path, index=False)
    summary.to_csv(summary_path, index=False)
    audit = summary[["method", "seed_count", "unique_sequences"]].copy()
    audit["independent_sequence_check"] = np.where(audit["unique_sequences"] == audit["seed_count"], "PASS", "REVIEW")
    audit.to_csv(audit_path, index=False)
    paired.to_csv(paired_path, index=False)
    write_table(summary, args.si_table)
    outputs = [per_seed_path, summary_path, audit_path, paired_path, args.si_table]
    pd.DataFrame(
        [{"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size} for path in outputs]
    ).to_csv(args.source_dir / "v60_same_protocol_baselines_output_sha256.csv", index=False)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
