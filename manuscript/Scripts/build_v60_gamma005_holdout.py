"""Audit and summarize the frozen gamma=0.05 held-out confirmation cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


SEEDS = tuple(range(15, 25))
METHODS = ("energy_gated_da_tpp", "predicted_target_greedy")
PROTOCOL_SHA256 = "1e1002fcd528df18a961d0c841db0c8b30222a46392ad9295a18b6b5f5d961fa"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_run(root: Path, method: str, seed: int) -> dict[str, object]:
    run = root / method / f"seed_{seed}" / "attempt_1"
    paths = {
        name: run / name
        for name in ("status.json", "run_config.json", "run_metrics.csv", "summary.csv")
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("; ".join(missing))
    status = json.loads(paths["status.json"].read_text(encoding="utf-8"))
    config = json.loads(paths["run_config.json"].read_text(encoding="utf-8"))
    if status.get("status") != "DONE":
        raise RuntimeError(f"run is not DONE: {run}")
    expected = {
        "method": method,
        "seed": seed,
        "gamma": 0.05,
        "formal_protocol_phase": "formal_evaluation",
        "formal_protocol_frozen": True,
        "formal_protocol_sha256": PROTOCOL_SHA256,
        "mc_passes": 30,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"{run}: {key}={config.get(key)!r}, expected {value!r}")
    metrics = pd.read_csv(paths["run_metrics.csv"])
    if len(metrics) != 1:
        raise ValueError(f"expected one metrics row: {paths['run_metrics.csv']}")
    metric = metrics.iloc[0]
    trajectory = pd.read_csv(paths["summary.csv"])
    if len(trajectory) != 40 or int(trajectory.iloc[-1]["oracle_evaluations"]) != 640:
        raise ValueError(f"incomplete trajectory: {paths['summary.csv']}")
    at_160 = trajectory.loc[trajectory["oracle_evaluations"].eq(160)]
    if len(at_160) != 1:
        raise ValueError(f"missing 160-query row: {paths['summary.csv']}")
    return {
        "method": method,
        "seed": seed,
        "AUTC_160": float(at_160.iloc[0]["AUTC_so_far"]),
        "AUTC_640": float(metric["AUTC"]),
        "recovery_at_80": int(metric["recovery_at_80"]),
        "recovery_at_160": int(metric["recovery_at_160"]),
        "recovery_at_240": int(metric["recovery_at_240"]),
        "recovery_at_320": int(metric["recovery_at_320"]),
        "final_recovery": int(metric["final_recovery"]),
        "direct_rounds": int(metric["direct_rounds"]),
        "correction_rounds": int(metric["correction_rounds"]),
        "total_correction_replacements": int(metric["total_correction_replacements"]),
        "total_correction_target_gain": int(metric["total_correction_target_gain"]),
        "sequence_sha256": str(metric["candidate_sequence_sha256"]),
        "run_config_sha256": sha256(paths["run_config.json"]),
        "run_metrics_sha256": sha256(paths["run_metrics.csv"]),
        "summary_sha256": sha256(paths["summary.csv"]),
        "run_path": str(run),
    }


def aggregate(per_seed: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    columns = (
        "AUTC_160",
        "AUTC_640",
        "recovery_at_80",
        "recovery_at_160",
        "recovery_at_240",
        "recovery_at_320",
        "final_recovery",
    )
    for method in METHODS:
        block = per_seed.loc[per_seed["method"].eq(method)]
        record: dict[str, object] = {
            "method": method,
            "seed_count": len(block),
            "unique_sequences": int(block["sequence_sha256"].nunique()),
        }
        for column in columns:
            record[f"mean_{column}"] = float(block[column].mean())
            record[f"sd_{column}"] = float(block[column].std(ddof=1))
        records.append(record)
    return pd.DataFrame(records)


def paired_analysis(per_seed: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    gate = per_seed.loc[per_seed["method"].eq(METHODS[0])].copy()
    greedy = per_seed.loc[per_seed["method"].eq(METHODS[1])].copy()
    paired = gate.merge(greedy, on="seed", suffixes=("_gate", "_greedy"), validate="one_to_one")
    metrics = ("AUTC_160", "AUTC_640", "recovery_at_80", "recovery_at_160")
    for metric in metrics:
        paired[f"delta_{metric}_gate_minus_greedy"] = paired[f"{metric}_gate"] - paired[f"{metric}_greedy"]
    delta = paired["delta_AUTC_160_gate_minus_greedy"].to_numpy(dtype=float)
    rng = np.random.default_rng(20260808)
    indices = rng.integers(0, len(delta), size=(100_000, len(delta)))
    boot = delta[indices].mean(axis=1)
    test = wilcoxon(
        delta,
        zero_method="wilcox",
        correction=False,
        alternative="two-sided",
        method="exact",
    )
    sd = float(np.std(delta, ddof=1))
    result: dict[str, object] = {
        "seed_count": len(delta),
        "mean_delta_AUTC_160_gate_minus_greedy": float(np.mean(delta)),
        "sd_delta_AUTC_160_gate_minus_greedy": sd,
        "bootstrap_seed": 20260808,
        "bootstrap_resamples": 100_000,
        "bootstrap_95ci_low": float(np.quantile(boot, 0.025)),
        "bootstrap_95ci_high": float(np.quantile(boot, 0.975)),
        "wilcoxon_statistic": float(test.statistic),
        "wilcoxon_exact_two_sided_p": float(test.pvalue),
        "paired_dz": float(np.mean(delta) / sd) if sd else None,
        "gate_wins_AUTC_160": int(np.sum(delta > 0)),
        "greedy_wins_AUTC_160": int(np.sum(delta < 0)),
        "ties_AUTC_160": int(np.sum(delta == 0)),
        "mean_delta_AUTC_640_gate_minus_greedy": float(
            paired["delta_AUTC_640_gate_minus_greedy"].mean()
        ),
        "mean_delta_recovery_at_80_gate_minus_greedy": float(
            paired["delta_recovery_at_80_gate_minus_greedy"].mean()
        ),
        "mean_delta_recovery_at_160_gate_minus_greedy": float(
            paired["delta_recovery_at_160_gate_minus_greedy"].mean()
        ),
    }
    paired["AUTC_160_leader"] = np.select(
        [paired["delta_AUTC_160_gate_minus_greedy"] > 0, paired["delta_AUTC_160_gate_minus_greedy"] < 0],
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
    return paired[keep].sort_values("seed"), result


def write_report(summary: pd.DataFrame, stats: dict[str, object], path: Path) -> None:
    gate = summary.loc[summary["method"].eq(METHODS[0])].iloc[0]
    greedy = summary.loc[summary["method"].eq(METHODS[1])].iloc[0]
    lines = [
        "# Gamma=0.05独立留出种子验证",
        "",
        "- 冻结协议：gamma=0.05，其余参数保持不变；正式留出seeds 15--24。",
        "- 运行状态：20/20 DONE；每种方法10条不同候选序列。",
        f"- AUTC_160：Gate {gate.mean_AUTC_160:.6f} ± {gate.sd_AUTC_160:.6f}；Greedy {greedy.mean_AUTC_160:.6f} ± {greedy.sd_AUTC_160:.6f}。",
        f"- 配对平均差：{stats['mean_delta_AUTC_160_gate_minus_greedy']:.6f}，100000次bootstrap 95% CI [{stats['bootstrap_95ci_low']:.6f}, {stats['bootstrap_95ci_high']:.6f}]。",
        f"- 配对胜负：Gate {stats['gate_wins_AUTC_160']}/10，Greedy {stats['greedy_wins_AUTC_160']}/10，平局 {stats['ties_AUTC_160']}/10。",
        f"- Exact Wilcoxon双侧p={stats['wilcoxon_exact_two_sided_p']:.6g}；dz={stats['paired_dz']:.4f}。",
        f"- Recovery@80：Gate {gate.mean_recovery_at_80:.2f}；Greedy {greedy.mean_recovery_at_80:.2f}。",
        f"- 完整AUTC：Gate {gate.mean_AUTC_640:.6f}；Greedy {greedy.mean_AUTC_640:.6f}。",
        "",
        "该报告仅描述预先冻结的留出确认结果；不得根据结果继续调整gamma或其他参数。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = [read_run(args.result_root, method, seed) for method in METHODS for seed in SEEDS]
    per_seed = pd.DataFrame(rows)
    summary = aggregate(per_seed)
    paired, stats = paired_analysis(per_seed)
    if not summary["unique_sequences"].eq(10).all():
        raise RuntimeError("held-out seed sequences are not independent within method")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "per_seed": args.output_dir / "v60_gamma005_holdout_per_seed.csv",
        "summary": args.output_dir / "v60_gamma005_holdout_summary.csv",
        "paired": args.output_dir / "v60_gamma005_holdout_paired.csv",
        "statistics": args.output_dir / "v60_gamma005_holdout_statistics.json",
        "report": args.output_dir / "V60_GAMMA005_HOLDOUT_REPORT_ZH.md",
    }
    per_seed.to_csv(outputs["per_seed"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    paired.to_csv(outputs["paired"], index=False)
    outputs["statistics"].write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    write_report(summary, stats, outputs["report"])
    manifest = pd.DataFrame(
        [{"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size} for path in outputs.values()]
    )
    manifest.to_csv(args.output_dir / "v60_gamma005_holdout_output_sha256.csv", index=False)
    print(summary.to_string(index=False))
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
