"""Score frozen CGCNN acquisition histories with a hidden DFT evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.three_system.retrospective_actual_dft import (
    build_hidden_evaluability_overlay,
    build_hidden_score_table,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS = ROOT / "results/three_system_low_data_v3_joint_endpoint"
DEFAULT_OUTPUT = ROOT / "results/cgcnn_hidden_evaluability_overlay_v1"
DEFAULT_SINGLE_RUN_ROOT = Path(
    r"D:\CGCNN\cloud_results\early_limo_hard640_budget640_20260618"
    r"\extracted\al_runs_early_limo_hard640_budget640_20260618\jobs"
)
CHECKPOINTS = (80, 160, 240, 320)
FORBIDDEN_ACQUISITION_COLUMNS = {
    "current_p_eval",
    "p_dft_evaluable",
    "predicted_p_dft_evaluable",
    "pseudo_dft_evaluable",
    "joint_qualified_probability",
}
PAIRED_METHODS = {
    "energy_gated_da_tpp": "Energy-Gated DA-TPP",
    "predicted_distance_greedy": "Predicted-Target Greedy",
}
SINGLE_RUN_METHODS = {
    "bs16_gated_ta_dpp": "Energy-Gated DA-TPP",
    "bs16_greedy": "Predicted-Target Greedy",
    "bs16_explore": "Explore",
    "bs16_modulus": "Modulus / Gradient-Norm Hybrid",
    "bs16_dropout": "MC Dropout",
    "bs16_random": "Random Sampling",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _history_hash(candidate_ids: pd.Series) -> str:
    payload = "\n".join(candidate_ids.astype(str)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_history(
    path: Path,
    *,
    method: str,
    seed: int,
    run_scope: str,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = {"id", "target_label"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    leaked = FORBIDDEN_ACQUISITION_COLUMNS & set(frame.columns)
    if leaked:
        raise ValueError(f"{path} leaks hidden DFT inputs: {sorted(leaked)}")
    if frame["id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate candidate IDs")

    result = pd.DataFrame(
        {
            "method": method,
            "seed": int(seed),
            "query": np.arange(1, len(frame) + 1, dtype=int),
            "candidate_id": frame["id"].astype(str),
            "target_label": pd.to_numeric(
                frame["target_label"], errors="raise"
            ).astype(int),
            "run_scope": run_scope,
            "source_history": str(path),
            "source_history_sha256": _sha256(path),
        }
    )
    if not result["target_label"].isin([0, 1]).all():
        raise ValueError(f"{path} has non-binary target labels")
    return result


def _paired_history_paths() -> list[tuple[Path, str, int]]:
    roots = {
        range(5, 10): (
            ROOT
            / "baseline_snapshot/archive/experiments/reproducibility/results"
            / "paired_two_dataset_confirmation_20260712/runs/limo"
        ),
        range(10, 15): (
            ROOT
            / "baseline_snapshot/archive/experiments/reproducibility/results"
            / "paired_two_dataset_confirmation_seeds_10_14_20260713"
            / "runs/limo"
        ),
    }
    paths: list[tuple[Path, str, int]] = []
    for seeds, base in roots.items():
        for seed in seeds:
            for directory, method in PAIRED_METHODS.items():
                paths.append(
                    (
                        base / directory / f"seed_{seed}/al_history.csv",
                        method,
                        seed,
                    )
                )
    return paths


def _load_paired_histories() -> tuple[pd.DataFrame, list[Path]]:
    frames: list[pd.DataFrame] = []
    inputs: list[Path] = []
    for path, method, seed in _paired_history_paths():
        if not path.is_file():
            raise FileNotFoundError(path)
        frames.append(
            _load_history(
                path,
                method=method,
                seed=seed,
                run_scope="formal_paired_seeds_5_14",
            )
        )
        inputs.append(path)
    return pd.concat(frames, ignore_index=True), inputs


def _load_single_run_histories(
    root: Path,
) -> tuple[pd.DataFrame, list[Path]]:
    frames: list[pd.DataFrame] = []
    inputs: list[Path] = []
    for directory, method in SINGLE_RUN_METHODS.items():
        path = root / directory / "al_history.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        frames.append(
            _load_history(
                path,
                method=method,
                seed=0,
                run_scope="legacy_single_run_illustrative_only",
            )
        )
        inputs.append(path)
    return pd.concat(frames, ignore_index=True), inputs


def _aggregate_overlay(results: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "target_hits",
        "expected_DFT_evaluable_target_hits",
        "ML_labeled_DFT_evaluable_target_hits",
        "mean_hidden_DFT_evaluable_probability_among_targets",
        "ML_labeled_DFT_evaluable_rate_among_targets",
    ]
    summary = (
        results.groupby(["method", "checkpoint"], sort=True)[metrics]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    summary.columns = [
        "_".join(column).rstrip("_") if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    summary["n_runs"] = (
        results.groupby(["method", "checkpoint"], sort=True)
        .size()
        .to_numpy()
    )
    return summary


def _paired_differences(results: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "target_hits",
        "expected_DFT_evaluable_target_hits",
        "ML_labeled_DFT_evaluable_target_hits",
        "mean_hidden_DFT_evaluable_probability_among_targets",
        "ML_labeled_DFT_evaluable_rate_among_targets",
    ]
    rows: list[dict[str, object]] = []
    gate_name = "Energy-Gated DA-TPP"
    greedy_name = "Predicted-Target Greedy"
    for checkpoint, checkpoint_rows in results.groupby("checkpoint", sort=True):
        indexed = checkpoint_rows.set_index(["seed", "method"])
        seeds = sorted(checkpoint_rows["seed"].unique())
        for metric in metrics:
            differences = np.asarray(
                [
                    float(indexed.loc[(seed, gate_name), metric])
                    - float(indexed.loc[(seed, greedy_name), metric])
                    for seed in seeds
                ],
                dtype=float,
            )
            rows.append(
                {
                    "checkpoint": int(checkpoint),
                    "metric": metric,
                    "n_pairs": len(differences),
                    "mean_Gate_minus_Greedy": float(differences.mean()),
                    "sample_sd": (
                        float(differences.std(ddof=1))
                        if len(differences) > 1
                        else np.nan
                    ),
                    "minimum": float(differences.min()),
                    "maximum": float(differences.max()),
                    "Gate_better_pairs": int((differences > 0).sum()),
                    "ties": int((differences == 0).sum()),
                    "Greedy_better_pairs": int((differences < 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def _target_set_audit(
    histories: pd.DataFrame,
    results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_lookup = results.set_index(["method", "seed", "checkpoint"])
    rows: list[dict[str, object]] = []
    for (method, seed), run in histories.groupby(["method", "seed"], sort=True):
        for checkpoint in CHECKPOINTS:
            prefix = run.loc[run["query"].le(checkpoint)]
            targets = sorted(
                prefix.loc[prefix["target_label"].eq(1), "candidate_id"].astype(str)
            )
            rows.append(
                {
                    "method": method,
                    "seed": int(seed),
                    "checkpoint": checkpoint,
                    "query_prefix_sha256": _history_hash(prefix["candidate_id"]),
                    "target_set_sha256": _history_hash(pd.Series(targets)),
                    "target_hits": int(
                        result_lookup.loc[
                            (method, int(seed), checkpoint), "target_hits"
                        ]
                    ),
                }
            )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["method", "checkpoint"], sort=True)
        .agg(
            n_seeds=("seed", "size"),
            unique_query_prefixes=("query_prefix_sha256", "nunique"),
            unique_target_sets=("target_set_sha256", "nunique"),
        )
        .reset_index()
    )
    return detail, summary


def _write_protocol(
    path: Path,
    *,
    selected_model: str,
    cv_row: pd.Series,
) -> None:
    text = f"""experiment_id: cgcnn_hidden_evaluability_overlay_v1
status: frozen_post_hoc_evaluation
acquisition:
  source: archived_original_CGCNN_active_learning_histories
  target_interval_eV_atom: [-2.18, -2.02]
  permitted_inputs:
    - CGCNN_target_interval_predictions
    - CGCNN_interval_hit_probability
    - CGCNN_uncertainty
    - candidate_embedding
    - composition_group_key
  forbidden_inputs:
    - DFT_evaluability_probability
    - DFT_evaluability_label
    - joint_target_and_DFT_score
hidden_evaluator:
  model: {selected_model}
  training_attempts: {int(cv_row['n'])}
  positive_attempts: {int(cv_row['positives'])}
  leave_one_out_ROC_AUC: {float(cv_row['loo_roc_auc']):.6f}
  leave_one_out_balanced_accuracy: {float(cv_row['loo_balanced_accuracy']):.6f}
  hard_label_threshold: 0.5
endpoint:
  primary: target_hits_scored_after_selection_by_hidden_DFT_evaluator
  expected_yield: sum_hidden_probability_over_target_hits
  hard_ML_label_yield: count_probability_at_least_0.5_over_target_hits
  observed_DFT_claim: false
checkpoints: [80, 160, 240, 320]
formal_comparison:
  methods: [Energy-Gated DA-TPP, Predicted-Target Greedy]
  paired_seeds: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
legacy_single_run_baselines:
  methods:
    - Energy-Gated DA-TPP
    - Predicted-Target Greedy
    - Explore
    - Modulus / Gradient-Norm Hybrid
    - MC Dropout
    - Random Sampling
  inferential_status: illustrative_only
"""
    path.write_text(text, encoding="utf-8")


def _markdown_table(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    digits: int = 2,
) -> list[str]:
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for row in frame.loc[:, columns].itertuples(index=False, name=None):
        rendered = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                rendered.append(f"{value:.{digits}f}")
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return lines


def _write_report(
    path: Path,
    *,
    cv_row: pd.Series,
    paired_summary: pd.DataFrame,
    paired_differences: pd.DataFrame,
    single_results: pd.DataFrame,
    target_set_summary: pd.DataFrame,
) -> None:
    paired_view = paired_summary.loc[
        :,
        [
            "method",
            "checkpoint",
            "target_hits_mean",
            "expected_DFT_evaluable_target_hits_mean",
            "ML_labeled_DFT_evaluable_target_hits_mean",
            "mean_hidden_DFT_evaluable_probability_among_targets_mean",
            "ML_labeled_DFT_evaluable_rate_among_targets_mean",
        ],
    ].rename(
        columns={
            "method": "方法",
            "checkpoint": "查询数",
            "target_hits_mean": "目标命中",
            "expected_DFT_evaluable_target_hits_mean": "预计可计算目标数",
            "ML_labeled_DFT_evaluable_target_hits_mean": "ML硬标签可计算目标数",
            "mean_hidden_DFT_evaluable_probability_among_targets_mean": (
                "目标内平均可计算概率"
            ),
            "ML_labeled_DFT_evaluable_rate_among_targets_mean": (
                "目标内ML可计算率"
            ),
        }
    )
    difference_view = paired_differences.loc[
        paired_differences["metric"].isin(
            [
                "expected_DFT_evaluable_target_hits",
                "ML_labeled_DFT_evaluable_target_hits",
                "mean_hidden_DFT_evaluable_probability_among_targets",
                "ML_labeled_DFT_evaluable_rate_among_targets",
            ]
        ),
        [
            "checkpoint",
            "metric",
            "mean_Gate_minus_Greedy",
            "Gate_better_pairs",
            "ties",
            "Greedy_better_pairs",
        ],
    ].rename(
        columns={
            "checkpoint": "查询数",
            "metric": "指标",
            "mean_Gate_minus_Greedy": "Gate减Greedy",
            "Gate_better_pairs": "Gate更高seed数",
            "ties": "相同seed数",
            "Greedy_better_pairs": "Greedy更高seed数",
        }
    )
    single_view = single_results.loc[
        single_results["checkpoint"].isin(CHECKPOINTS),
        [
            "method",
            "checkpoint",
            "target_hits",
            "expected_DFT_evaluable_target_hits",
            "ML_labeled_DFT_evaluable_target_hits",
            "mean_hidden_DFT_evaluable_probability_among_targets",
            "ML_labeled_DFT_evaluable_rate_among_targets",
        ],
    ].rename(
        columns={
            "method": "方法",
            "checkpoint": "查询数",
            "target_hits": "目标命中",
            "expected_DFT_evaluable_target_hits": "预计可计算目标数",
            "ML_labeled_DFT_evaluable_target_hits": "ML硬标签可计算目标数",
            "mean_hidden_DFT_evaluable_probability_among_targets": (
                "目标内平均可计算概率"
            ),
            "ML_labeled_DFT_evaluable_rate_among_targets": "目标内ML可计算率",
        }
    )
    identity = target_set_summary.rename(
        columns={
            "method": "方法",
            "checkpoint": "查询数",
            "n_seeds": "seed数",
            "unique_query_prefixes": "不同查询前缀数",
            "unique_target_sets": "不同目标集合数",
        }
    )

    lines = [
        "# CGCNN目标检索轨迹的隐藏DFT可评价性审计",
        "",
        "## 结论",
        "",
        "本次纠正后的实验中，所有采集策略只使用原始CGCNN目标区间预测、"
        "命中概率、不确定性和原有多样性信息。DFT可评价性模型不参与候选"
        "选择，只在候选已经被原始策略选出后进行隐藏评分。",
        "",
        "因此，Predicted-Target Greedy没有获得DFT可评价性标签或概率。此前"
        "按该概率排序的`DFT-evaluable Greedy`与`Joint-qualified Greedy`"
        "属于另一个实验问题，不进入本次主比较。",
        "",
        "## 隐藏评价模型",
        "",
        f"- 模型：`{cv_row['model_name']}`。",
        f"- 历史DFT尝试：{int(cv_row['n'])}个，其中严格可评价"
        f"{int(cv_row['positives'])}个。",
        f"- 留一交叉验证ROC-AUC：{float(cv_row['loo_roc_auc']):.3f}；"
        f"平衡准确率：{float(cv_row['loo_balanced_accuracy']):.3f}。",
        "- 历史训练候选使用OOF概率，其余候选使用冻结全模型概率。",
        "- 这些结果是ML估计的DFT可评价性，不是新增VASP观测。",
        "",
        "## 十个配对seed的正式叠加结果",
        "",
        *_markdown_table(paired_view, list(paired_view.columns)),
        "",
        "## Gate相对Greedy的配对差值",
        "",
        *_markdown_table(difference_view, list(difference_view.columns)),
        "",
        "正差表示Gate在相同查询预算下更早找到更多“目标区间命中且被隐藏"
        "模型评为DFT可评价”的候选。该优势集中在早期预算；到240次查询时"
        "Greedy追平或略微反超，到320次两者检完同一批78个目标后完全相同。",
        "",
        "## seed独立性审计",
        "",
        *_markdown_table(identity, list(identity.columns), digits=0),
        "",
        "审计显示，80、160和240次查询时，每种方法的10个归档seed连完整"
        "查询前缀都相同；到320次时非目标候选顺序开始变化，但目标集合仍"
        "完全相同。因此早期差值只有一个独立轨迹，不能写成10次独立重复，"
        "也不能据此计算有意义的配对显著性。",
        "",
        "## 原六基线单次历史轨迹",
        "",
        "这些是旧的单次完整六策略轨迹，仅作描述性补充，不能替代十seed"
        "Gate–Greedy正式比较。",
        "",
        *_markdown_table(single_view, list(single_view.columns)),
        "",
        "## 论文使用边界",
        "",
        "- 可以报告：在旧CGCNN目标区间任务下，Gate早期找到的目标候选中，"
        "隐藏ML评价器预测为DFT可评价的数量更多。",
        "- 不可以报告：这些候选已经获得了新的真实DFT验证。",
        "- 不可以把DFT可评价性概率输入任何采集策略后，仍称为同一实验。",
        "- 新的VASP前瞻验证若以后完成，应单独作为真实高保真证据。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_manifest(paths: list[Path], output: Path) -> None:
    rows = [
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(set(paths), key=lambda item: str(item).lower())
    ]
    pd.DataFrame(rows).to_csv(output, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--single-run-root",
        type=Path,
        default=DEFAULT_SINGLE_RUN_ROOT,
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    full_scores_path = args.analysis_dir / "three_system_candidate_scores.csv"
    oof_scores_path = (
        args.analysis_dir / "dft_evaluability_model_oof_predictions.csv"
    )
    cv_path = args.analysis_dir / "dft_evaluability_model_cv.csv"

    full_scores = pd.read_csv(full_scores_path)
    models = full_scores["dft_evaluability_model"].dropna().unique()
    if len(models) != 1:
        raise ValueError(f"expected one frozen evaluator model, found {models}")
    selected_model = str(models[0])
    oof_scores = pd.read_csv(oof_scores_path)
    hidden_scores = build_hidden_score_table(
        full_scores,
        oof_scores,
        selected_model=selected_model,
    )
    cv = pd.read_csv(cv_path)
    cv_rows = cv.loc[cv["model_name"].astype(str).eq(selected_model)]
    if len(cv_rows) != 1:
        raise ValueError(f"missing unique CV row for {selected_model}")
    cv_row = cv_rows.iloc[0]

    paired_histories, paired_inputs = _load_paired_histories()
    paired_detail, paired_results = build_hidden_evaluability_overlay(
        paired_histories,
        hidden_scores,
        checkpoints=CHECKPOINTS,
    )
    paired_summary = _aggregate_overlay(paired_results)
    paired_difference = _paired_differences(paired_results)
    target_audit, target_summary = _target_set_audit(
        paired_histories,
        paired_results,
    )

    single_histories, single_inputs = _load_single_run_histories(
        args.single_run_root
    )
    single_detail, single_results = build_hidden_evaluability_overlay(
        single_histories,
        hidden_scores,
        checkpoints=CHECKPOINTS,
    )

    outputs = {
        "hidden_dft_evaluability_scores.csv": hidden_scores,
        "paired_cgcnn_histories.csv": paired_histories,
        "paired_hidden_evaluability_detail.csv": paired_detail,
        "paired_hidden_evaluability_results.csv": paired_results,
        "paired_hidden_evaluability_summary.csv": paired_summary,
        "paired_gate_minus_greedy.csv": paired_difference,
        "single_run_baseline_histories.csv": single_histories,
        "single_run_hidden_evaluability_detail.csv": single_detail,
        "single_run_hidden_evaluability_results.csv": single_results,
        "target_set_identity_audit.csv": target_audit,
        "target_set_identity_summary.csv": target_summary,
    }
    written: list[Path] = []
    for filename, frame in outputs.items():
        path = args.output_dir / filename
        frame.to_csv(path, index=False)
        written.append(path)

    protocol_path = args.output_dir / "CGCNN_HIDDEN_EVALUATOR_PROTOCOL.yaml"
    _write_protocol(protocol_path, selected_model=selected_model, cv_row=cv_row)
    written.append(protocol_path)
    report_path = args.output_dir / "CGCNN_HIDDEN_EVALUABILITY_REPORT.md"
    _write_report(
        report_path,
        cv_row=cv_row,
        paired_summary=paired_summary,
        paired_differences=paired_difference,
        single_results=single_results,
        target_set_summary=target_summary,
    )
    written.append(report_path)

    metadata_path = args.output_dir / "run_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "python": sys.version,
                "platform": platform.platform(),
                "pandas": pd.__version__,
                "numpy": np.__version__,
                "selected_hidden_evaluator": selected_model,
                "formal_paired_runs": 20,
                "paired_seeds": list(range(5, 15)),
                "checkpoints": list(CHECKPOINTS),
                "new_DFT_runs": 0,
                "new_CGCNN_runs": 0,
                "evidence_tier": (
                    "post_selection_hidden_ML_evaluator_not_observed_DFT"
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    written.append(metadata_path)

    input_manifest = args.output_dir / "INPUT_SHA256_MANIFEST.csv"
    _write_manifest(
        [
            full_scores_path,
            oof_scores_path,
            cv_path,
            *paired_inputs,
            *single_inputs,
        ],
        input_manifest,
    )
    written.append(input_manifest)
    _write_manifest(written, args.output_dir / "OUTPUT_SHA256_MANIFEST.csv")


if __name__ == "__main__":
    main()
