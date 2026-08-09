"""Build manuscript-facing tables and figures from verified evidence files."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


OBSERVED_DFT_TIER = "observed_DFT_retrospective_selected_subset"
PDF_METADATA = {
    "Creator": "Energy-Gated DA-TPP manuscript asset builder",
    "CreationDate": None,
    "ModDate": None,
}


def build_actual_dft_table(
    summary: pd.DataFrame,
    curve: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize only the observed retrospective DFT selected subset."""
    summary_required = {
        "policy",
        "final_DFT_evaluable",
        "time_to_first_evaluable_round",
        "DFT_evaluable_round_AUC",
        "evidence_tier",
    }
    curve_required = {
        "policy",
        "round",
        "cumulative_DFT_evaluable",
        "evidence_tier",
    }
    summary_missing = summary_required - set(summary.columns)
    curve_missing = curve_required - set(curve.columns)
    if summary_missing or curve_missing:
        raise ValueError(
            "observed DFT inputs are missing columns: "
            f"summary={sorted(summary_missing)}, curve={sorted(curve_missing)}"
        )
    if not summary["evidence_tier"].eq(OBSERVED_DFT_TIER).all():
        raise ValueError("summary contains non-observed evidence")
    if not curve["evidence_tier"].eq(OBSERVED_DFT_TIER).all():
        raise ValueError("curve contains non-observed evidence")

    round_12 = curve.loc[
        curve["round"].astype(int) == 12,
        ["policy", "cumulative_DFT_evaluable"],
    ].rename(columns={"cumulative_DFT_evaluable": "round_12_evaluable"})
    if set(round_12["policy"]) != set(summary["policy"]):
        raise ValueError("round 12 is missing for one or more policies")
    table = summary.loc[
        :,
        [
            "policy",
            "time_to_first_evaluable_round",
            "final_DFT_evaluable",
            "DFT_evaluable_round_AUC",
            "evidence_tier",
        ],
    ].merge(round_12, on="policy", validate="one_to_one")
    table = table.rename(
        columns={
            "time_to_first_evaluable_round": "first_evaluable_round",
            "final_DFT_evaluable": "final_evaluable",
            "DFT_evaluable_round_AUC": "count_round_auc",
        }
    )
    return table.loc[
        :,
        [
            "policy",
            "first_evaluable_round",
            "round_12_evaluable",
            "final_evaluable",
            "count_round_auc",
            "evidence_tier",
        ],
    ]


def build_model_validation_table(
    binary_cv: pd.DataFrame,
    energy_cv: pd.DataFrame,
) -> pd.DataFrame:
    """Keep binary evaluability and continuous-energy metrics distinct."""
    binary_required = {
        "model_name",
        "n",
        "loo_roc_auc",
        "loo_balanced_accuracy",
        "loo_brier_score",
        "loo_log_loss",
    }
    energy_required = {
        "model_id",
        "n",
        "loo_mae_eV_atom",
        "loo_rmse_eV_atom",
        "prediction_interval_coverage_95",
    }
    binary_missing = binary_required - set(binary_cv.columns)
    energy_missing = energy_required - set(energy_cv.columns)
    if binary_missing or energy_missing:
        raise ValueError(
            "model validation inputs are missing columns: "
            f"binary={sorted(binary_missing)}, energy={sorted(energy_missing)}"
        )

    binary = pd.DataFrame(
        {
            "task": "DFT evaluability",
            "model": binary_cv["model_name"],
            "n": binary_cv["n"],
            "roc_auc": binary_cv["loo_roc_auc"],
            "balanced_accuracy": binary_cv["loo_balanced_accuracy"],
            "brier_score": binary_cv["loo_brier_score"],
            "log_loss": binary_cv["loo_log_loss"],
            "mae_eV_atom": pd.NA,
            "rmse_eV_atom": pd.NA,
            "prediction_interval_coverage_95": pd.NA,
            "evidence_tier": "observed_DFT_nested_LOO",
        }
    )
    energy = pd.DataFrame(
        {
            "task": "DFT formation-energy calibration",
            "model": energy_cv["model_id"],
            "n": energy_cv["n"],
            "roc_auc": pd.NA,
            "balanced_accuracy": pd.NA,
            "brier_score": pd.NA,
            "log_loss": pd.NA,
            "mae_eV_atom": energy_cv["loo_mae_eV_atom"],
            "rmse_eV_atom": energy_cv["loo_rmse_eV_atom"],
            "prediction_interval_coverage_95": energy_cv[
                "prediction_interval_coverage_95"
            ],
            "evidence_tier": "observed_DFT_energy_LOO",
        }
    )
    return pd.concat([binary, energy], ignore_index=True)


def build_expected_yield_table(
    replay_results: pd.DataFrame,
    *,
    checkpoint: int = 32,
    equivalent_aliases: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Aggregate paired replay estimates without double-counting aliases."""
    required = {
        "method",
        "seed",
        "checkpoint",
        "estimated_interval_hit_count",
        "simulated_interval_hit_count",
        "estimated_DFT_evaluable_count",
        "unique_structure_clusters",
        "evidence_tier",
    }
    missing = required - set(replay_results.columns)
    if missing:
        raise ValueError(f"replay results are missing columns: {sorted(missing)}")

    frame = replay_results.loc[
        replay_results["checkpoint"].astype(int) == int(checkpoint)
    ].copy()
    if frame.empty:
        raise ValueError(f"checkpoint {checkpoint} is absent from replay results")
    aliases = equivalent_aliases or {}
    aliases_by_canonical: dict[str, list[str]] = {}
    comparison_columns = [
        "seed",
        "estimated_interval_hit_count",
        "simulated_interval_hit_count",
        "estimated_DFT_evaluable_count",
        "unique_structure_clusters",
        "evidence_tier",
    ]
    for alias, canonical in aliases.items():
        alias_rows = frame.loc[
            frame["method"] == alias, comparison_columns
        ].sort_values("seed", kind="mergesort").reset_index(drop=True)
        canonical_rows = frame.loc[
            frame["method"] == canonical, comparison_columns
        ].sort_values("seed", kind="mergesort").reset_index(drop=True)
        if alias_rows.empty or canonical_rows.empty:
            raise ValueError(f"missing alias pair: {alias} -> {canonical}")
        pd.testing.assert_frame_equal(alias_rows, canonical_rows)
        frame = frame.loc[frame["method"] != alias]
        aliases_by_canonical.setdefault(canonical, []).append(alias)

    tier_counts = frame.groupby("method")["evidence_tier"].nunique()
    if not tier_counts.eq(1).all():
        raise ValueError("a policy contains multiple evidence tiers")
    table = (
        frame.groupby("method", as_index=False)
        .agg(
            mean_ml_labeled_qualified=("simulated_interval_hit_count", "mean"),
            sd_ml_labeled_qualified=("simulated_interval_hit_count", "std"),
            mean_expected_qualified=("estimated_interval_hit_count", "mean"),
            sd_expected_qualified=("estimated_interval_hit_count", "std"),
            mean_expected_evaluable=("estimated_DFT_evaluable_count", "mean"),
            mean_structure_clusters=("unique_structure_clusters", "mean"),
            evidence_tier=("evidence_tier", "first"),
        )
        .sort_values(
            ["mean_ml_labeled_qualified", "mean_expected_qualified", "method"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    table["checkpoint"] = int(checkpoint)
    table["equivalent_aliases"] = table["method"].map(
        lambda method: "|".join(sorted(aliases_by_canonical.get(method, [])))
    )
    return table.loc[
        :,
        [
            "method",
            "checkpoint",
            "mean_ml_labeled_qualified",
            "sd_ml_labeled_qualified",
            "mean_expected_qualified",
            "sd_expected_qualified",
            "mean_expected_evaluable",
            "mean_structure_clusters",
            "equivalent_aliases",
            "evidence_tier",
        ],
    ]


def _latex_escape(value: object) -> str:
    text = str(value)
    for old, new in (
        ("\\", r"\textbackslash{}"),
        ("_", r"\_"),
        ("%", r"\%"),
        ("&", r"\&"),
        ("#", r"\#"),
    ):
        text = text.replace(old, new)
    return text


def _write_model_table(path: Path, table: pd.DataFrame) -> None:
    binary = table.loc[table["task"] == "DFT evaluability"].iloc[0]
    energy = table.loc[
        table["task"] == "DFT formation-energy calibration"
    ].iloc[0]
    lines = [
        r"\begin{tabular}{@{}llrrrr@{}}",
        r"\toprule",
        r"Task & Selected model & $n$ & Primary metric & Secondary metric & Coverage \\",
        r"\midrule",
        (
            f"DFT evaluability & {_latex_escape(binary['model'])} & "
            f"{int(binary['n'])} & ROC--AUC {float(binary['roc_auc']):.3f} & "
            f"Brier {float(binary['brier_score']):.3f} & -- \\\\"
        ),
        (
            f"Energy calibration & {_latex_escape(energy['model'])} & "
            f"{int(energy['n'])} & MAE {float(energy['mae_eV_atom']):.4f} & "
            f"RMSE {float(energy['rmse_eV_atom']):.4f} & "
            f"{100 * float(energy['prediction_interval_coverage_95']):.0f}\\% \\\\"
        ),
        r"\bottomrule",
        r"\end{tabular}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _display_method(method: str, aliases: str = "") -> str:
    labels = {
        "group_gated_da_tpp": "Group-Gated DA-TPP",
        "group_only": "Group-only",
        "always_correction": "Always-correction",
        "margin_only": "Margin-only",
        "dft_evaluable_greedy": "DFT-evaluable Greedy",
        "joint_qualified_greedy": "Joint-qualified Greedy",
        "predicted_target_greedy": "Predicted-target Greedy",
        "explore_core_set": "Explore/core-set",
        "gradient_norm_hybrid": "Gradient-norm hybrid",
        "mlip_energy_greedy": "MLIP-energy Greedy",
        "composition_only": "Composition-only",
        "mc_uncertainty_only": "Uncertainty-only",
        "random_sampling": "Random",
    }
    label = labels.get(method, method.replace("_", " "))
    if aliases:
        alias_labels = [labels.get(item, item.replace("_", " ")) for item in aliases.split("|")]
        label = f"{label} (same selector as {', '.join(alias_labels)})"
    return label


def _write_expected_yield_table(path: Path, table: pd.DataFrame) -> None:
    lines = [
        r"\begin{tabular}{@{}lrrr@{}}",
        r"\toprule",
        r"Policy & ML-labeled qualified count & Expected qualified yield & Structure clusters \\",
        r"\midrule",
    ]
    for row in table.itertuples(index=False):
        lines.append(
            f"{_latex_escape(_display_method(row.method, row.equivalent_aliases))} & "
            f"{row.mean_ml_labeled_qualified:.2f} $\\pm$ "
            f"{row.sd_ml_labeled_qualified:.2f} & "
            f"{row.mean_expected_qualified:.2f} $\\pm$ "
            f"{row.sd_expected_qualified:.2f} & "
            f"{row.mean_structure_clusters:.1f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _write_actual_dft_table(path: Path, table: pd.DataFrame) -> None:
    lines = [
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"Policy & First evaluable round & Evaluable at round 12 & Final evaluable & Count--round AUC \\",
        r"\midrule",
    ]
    for row in table.itertuples(index=False):
        lines.append(
            f"{_latex_escape(row.policy)} & {int(row.first_evaluable_round)} & "
            f"{int(row.round_12_evaluable)} & {int(row.final_evaluable)} & "
            f"{float(row.count_round_auc):.1f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _write_macros(
    path: Path,
    *,
    pool_count: int,
    pool_counts_by_element: dict[str, int],
    history_count: int,
    positive_count: int,
    actual_subset_count: int,
    actual_failure_count: int,
    binary_row: pd.Series,
    energy_row: pd.Series,
    expected_table: pd.DataFrame,
    actual_table: pd.DataFrame,
) -> None:
    group = expected_table.set_index("method").loc["group_gated_da_tpp"]
    dft_greedy = expected_table.set_index("method").loc[
        "dft_evaluable_greedy"
    ]
    target_greedy = expected_table.set_index("method").loc[
        "joint_qualified_greedy"
    ]
    actual = actual_table.set_index("policy")
    values = {
        "MFPoolCount": f"{pool_count:d}",
        "MFCrCount": f"{pool_counts_by_element.get('Cr', 0):d}",
        "MFMnCount": f"{pool_counts_by_element.get('Mn', 0):d}",
        "MFMgCount": f"{pool_counts_by_element.get('Mg', 0):d}",
        "MFHistoryCount": f"{history_count:d}",
        "MFProspectiveCount": f"{pool_count - history_count:d}",
        "MFPositiveCount": f"{positive_count:d}",
        "MFEnergyCount": f"{int(energy_row['n']):d}",
        "EvalCVROC": f"{float(binary_row['loo_roc_auc']):.3f}",
        "EvalCVBalancedAccuracy": (
            f"{float(binary_row['loo_balanced_accuracy']):.3f}"
        ),
        "EvalCVBrier": f"{float(binary_row['loo_brier_score']):.3f}",
        "EvalCVLogLoss": f"{float(binary_row['loo_log_loss']):.3f}",
        "EnergyCVMAE": f"{float(energy_row['loo_mae_eV_atom']):.4f}",
        "EnergyCVRMSE": f"{float(energy_row['loo_rmse_eV_atom']):.4f}",
        "EnergyCVCoveragePercent": (
            f"{100 * float(energy_row['prediction_interval_coverage_95']):.0f}"
        ),
        "MFGroupQualifiedMean": f"{group['mean_expected_qualified']:.2f}",
        "MFGroupQualifiedSD": f"{group['sd_expected_qualified']:.2f}",
        "MFGroupMLLabeledQualifiedMean": (
            f"{group['mean_ml_labeled_qualified']:.2f}"
        ),
        "MFGroupMLLabeledQualifiedSD": (
            f"{group['sd_ml_labeled_qualified']:.2f}"
        ),
        "MFGroupEvaluableMean": f"{group['mean_expected_evaluable']:.2f}",
        "MFDFTGreedyQualifiedMean": (
            f"{dft_greedy['mean_expected_qualified']:.2f}"
        ),
        "MFDFTGreedyEvaluableMean": (
            f"{dft_greedy['mean_expected_evaluable']:.2f}"
        ),
        "MFQualifiedGainOverDFTGreedy": (
            f"{group['mean_expected_qualified'] - dft_greedy['mean_expected_qualified']:.2f}"
        ),
        "MFDFTGreedyMLLabeledQualifiedMean": (
            f"{dft_greedy['mean_ml_labeled_qualified']:.2f}"
        ),
        "MFDFTGreedyMLLabeledQualifiedSD": (
            f"{dft_greedy['sd_ml_labeled_qualified']:.2f}"
        ),
        "MFJointGreedyMLLabeledQualifiedMean": (
            f"{target_greedy['mean_ml_labeled_qualified']:.2f}"
        ),
        "MFJointGreedyMLLabeledQualifiedSD": (
            f"{target_greedy['sd_ml_labeled_qualified']:.2f}"
        ),
        "MFMLLabeledGainOverDFTGreedy": (
            f"{group['mean_ml_labeled_qualified'] - dft_greedy['mean_ml_labeled_qualified']:.2f}"
        ),
        "MFMLLabeledGainOverJointGreedy": (
            f"{group['mean_ml_labeled_qualified'] - target_greedy['mean_ml_labeled_qualified']:.2f}"
        ),
        "ActualGateRoundTwelve": (
            f"{int(actual.loc['Gate', 'round_12_evaluable'])}"
        ),
        "ActualGreedyRoundTwelve": (
            f"{int(actual.loc['Greedy', 'round_12_evaluable'])}"
        ),
        "ActualGateAUC": f"{float(actual.loc['Gate', 'count_round_auc']):.0f}",
        "ActualGreedyAUC": (
            f"{float(actual.loc['Greedy', 'count_round_auc']):.0f}"
        ),
        "ActualSubsetCount": f"{actual_subset_count:d}",
        "ActualFinalEvaluable": (
            f"{int(actual.loc['Gate', 'final_evaluable'])}"
        ),
        "ActualFailureCount": f"{actual_failure_count:d}",
    }
    lines = [
        f"\\newcommand{{\\{name}}}{{{value}}}"
        for name, value in values.items()
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _audit_selector_aliases(
    selections: pd.DataFrame,
    aliases: dict[str, str],
) -> pd.DataFrame:
    required = {
        "method",
        "seed",
        "query",
        "candidate_id",
        "proxy_model_seed",
    }
    missing = required - set(selections.columns)
    if missing:
        raise ValueError(
            f"selection records are missing columns: {sorted(missing)}"
        )
    rows: list[dict[str, object]] = []
    comparison_columns = [
        "seed",
        "query",
        "candidate_id",
        "proxy_model_seed",
    ]
    for alias, canonical in aliases.items():
        alias_rows = (
            selections.loc[
                selections["method"] == alias,
                comparison_columns,
            ]
            .sort_values(["seed", "query"], kind="mergesort")
            .reset_index(drop=True)
        )
        canonical_rows = (
            selections.loc[
                selections["method"] == canonical,
                comparison_columns,
            ]
            .sort_values(["seed", "query"], kind="mergesort")
            .reset_index(drop=True)
        )
        equal = not alias_rows.empty and alias_rows.equals(canonical_rows)
        rows.append(
            {
                "canonical_method": canonical,
                "equivalent_alias": alias,
                "same_complete_selection_record": bool(equal),
            }
        )
        if not equal:
            raise ValueError(
                f"selector alias audit failed: {alias} -> {canonical}"
            )
    return pd.DataFrame(rows)


def build_manuscript_assets(
    analysis_dir: str | Path,
    observed_dir: str | Path,
    output_dir: str | Path,
) -> None:
    """Generate deterministic multi-fidelity manuscript assets."""
    analysis_dir = Path(analysis_dir)
    observed_dir = Path(observed_dir)
    output_dir = Path(output_dir)
    figures_dir = output_dir / "Figures"
    tables_dir = output_dir / "Tables/generated"
    source_dir = output_dir / "SourceData"
    for directory in (figures_dir, tables_dir, source_dir):
        directory.mkdir(parents=True, exist_ok=True)

    binary_cv = pd.read_csv(analysis_dir / "dft_evaluability_model_cv.csv")
    energy_cv = pd.read_csv(analysis_dir / "formation_energy_calibration_cv.csv")
    energy_oof = pd.read_csv(
        analysis_dir / "formation_energy_calibration_oof_predictions.csv"
    )
    replay = pd.read_csv(analysis_dir / "paired_baseline_replay_results.csv")
    selections = pd.read_csv(
        analysis_dir / "paired_baseline_replay_selections.csv"
    )
    equivalence = pd.read_csv(
        analysis_dir / "paired_selector_equivalence_audit.csv"
    )
    pool = pd.read_csv(analysis_dir / "three_system_pool.csv")
    labels = pd.read_csv(analysis_dir / "historical_dft_binary_labels.csv")
    observed_summary = pd.read_csv(
        observed_dir / "actual_dft_gate_greedy_summary.csv"
    )
    observed_curve = pd.read_csv(
        observed_dir / "actual_dft_gate_greedy_curve.csv"
    )

    equivalence_ok = (
        equivalence["same_candidate_sequence"].astype(str).str.lower().eq("true")
        & equivalence["same_proxy_model_seed_sequence"]
        .astype(str)
        .str.lower()
        .eq("true")
    )
    if not equivalence_ok.all():
        raise ValueError("paired selector equivalence audit did not pass")
    aliases = {
        "full_gate": "group_gated_da_tpp",
        "explore_core_set": "always_correction",
    }
    selector_alias_audit = _audit_selector_aliases(selections, aliases)

    model_table = build_model_validation_table(binary_cv, energy_cv)
    expected_table = build_expected_yield_table(
        replay,
        checkpoint=32,
        equivalent_aliases=aliases,
    )
    actual_table = build_actual_dft_table(observed_summary, observed_curve)

    canonical_replay = replay.loc[~replay["method"].isin(aliases)].copy()
    trajectories = (
        canonical_replay.groupby(["method", "checkpoint"], as_index=False)
        .agg(
            mean_ml_labeled_qualified=("simulated_interval_hit_count", "mean"),
            sd_ml_labeled_qualified=("simulated_interval_hit_count", "std"),
            mean_expected_qualified=("estimated_interval_hit_count", "mean"),
            sd_expected_qualified=("estimated_interval_hit_count", "std"),
            mean_expected_evaluable=("estimated_DFT_evaluable_count", "mean"),
            sd_expected_evaluable=("estimated_DFT_evaluable_count", "std"),
            evidence_tier=("evidence_tier", "first"),
        )
        .sort_values(["method", "checkpoint"], kind="mergesort")
    )

    best_energy_id = str(energy_cv.iloc[0]["model_id"])
    best_energy_oof = energy_oof.loc[
        energy_oof["model_id"] == best_energy_id
    ].copy()
    model_table.to_csv(source_dir / "model_validation.csv", index=False)
    best_energy_oof.to_csv(
        source_dir / "energy_calibration_best_oof.csv", index=False
    )
    expected_table.to_csv(
        source_dir / "multifidelity_expected_yield.csv", index=False
    )
    trajectories.to_csv(
        source_dir / "multifidelity_expected_yield_trajectories.csv",
        index=False,
    )
    actual_table.to_csv(source_dir / "actual_dft_ordering.csv", index=False)
    observed_curve.to_csv(source_dir / "actual_dft_curve.csv", index=False)
    selector_alias_audit.to_csv(
        source_dir / "selector_equivalence_audit.csv",
        index=False,
    )

    _write_model_table(tables_dir / "table_model_validation.tex", model_table)
    _write_expected_yield_table(
        tables_dir / "table_expected_yield.tex", expected_table
    )
    _write_actual_dft_table(
        tables_dir / "table_actual_dft_ordering.tex", actual_table
    )
    _write_macros(
        tables_dir / "multifidelity_metrics_macros.tex",
        pool_count=len(pool),
        pool_counts_by_element={
            str(element): int(count)
            for element, count in pool["m_element"].value_counts().items()
        },
        history_count=len(labels),
        positive_count=int(labels["dft_evaluable"].astype(int).sum()),
        actual_subset_count=int(
            observed_summary["n_observed_candidates"].astype(int).iloc[0]
        ),
        actual_failure_count=int(
            observed_summary["final_DFT_failures"].astype(int).iloc[0]
        ),
        binary_row=binary_cv.iloc[0],
        energy_row=energy_cv.iloc[0],
        expected_table=expected_table,
        actual_table=actual_table,
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 6.5,
            "pdf.fonttype": 42,
        }
    )

    figure, axis = plt.subplots(figsize=(7.2, 3.15))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    boxes = [
        (
            0.02,
            0.58,
            0.18,
            0.25,
            "Fixed Cr/Mn/Mg pool\n275 candidates",
            "#D9EAF7",
        ),
        (
            0.27,
            0.58,
            0.19,
            0.25,
            "Low-fidelity descriptors\nCHGNet + MACE-MP",
            "#E8E3F3",
        ),
        (
            0.53,
            0.58,
            0.19,
            0.25,
            "Low-data calibration\n20 binary / 10 energy",
            "#FBE5D6",
        ),
        (
            0.79,
            0.58,
            0.19,
            0.25,
            "Paired policy replay\nML-labeled target count",
            "#E2F0D9",
        ),
    ]
    for x, y, width, height, text, color in boxes:
        patch = plt.Rectangle(
            (x, y),
            width,
            height,
            facecolor=color,
            edgecolor="0.35",
            linewidth=0.9,
        )
        axis.add_patch(patch)
        axis.text(
            x + width / 2,
            y + height / 2,
            text,
            ha="center",
            va="center",
            fontsize=8,
        )
    for left, right in zip(boxes[:-1], boxes[1:]):
        axis.annotate(
            "",
            xy=(right[0] - 0.01, right[1] + right[3] / 2),
            xytext=(left[0] + left[2] + 0.01, left[1] + left[3] / 2),
            arrowprops={"arrowstyle": "->", "color": "0.35", "linewidth": 1.0},
        )

    observed_box = plt.Rectangle(
        (0.53, 0.12),
        0.45,
        0.22,
        facecolor="#FFF2CC",
        edgecolor="0.35",
        linewidth=0.9,
    )
    axis.add_patch(observed_box)
    axis.text(
        0.755,
        0.23,
        "Observed DFT check: retrospective selected subset\n"
        "12 attempted candidates; ordering evidence only",
        ha="center",
        va="center",
        fontsize=8,
    )
    axis.annotate(
        "",
        xy=(0.755, 0.35),
        xytext=(0.885, 0.57),
        arrowprops={"arrowstyle": "->", "color": "0.35", "linewidth": 1.0},
    )
    axis.text(
        0.02,
        0.23,
        "Evidence boundary",
        fontsize=8,
        fontweight="bold",
        color="#9C6500",
    )
    axis.text(
        0.02,
        0.14,
        "Replay outputs are model estimates;\n"
        "only completed VASP records are observed DFT.",
        fontsize=7.5,
        va="center",
    )
    figure.tight_layout()
    figure.savefig(
        figures_dir / "Figure1_multifidelity_workflow.pdf",
        bbox_inches="tight",
        metadata=PDF_METADATA,
    )
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    binary_plot = binary_cv.sort_values("loo_roc_auc", ascending=True)
    axes[0].barh(
        binary_plot["model_name"].map(lambda value: value.replace("_", " ")),
        binary_plot["loo_roc_auc"],
        color="#4C78A8",
    )
    axes[0].axvline(0.5, color="0.4", linestyle="--", linewidth=0.8)
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("Nested leave-one-out ROC–AUC")
    axes[0].set_title("(a) DFT-evaluability model")
    observed = best_energy_oof["observed_dft_energy_eV_atom"].astype(float)
    predicted = best_energy_oof["predicted_dft_energy_eV_atom"].astype(float)
    axes[1].scatter(observed, predicted, color="#F58518", s=24, zorder=3)
    low = float(min(observed.min(), predicted.min()))
    high = float(max(observed.max(), predicted.max()))
    pad = max(0.02, 0.05 * (high - low if high > low else 1.0))
    axes[1].plot([low - pad, high + pad], [low - pad, high + pad], "--", color="0.4")
    axes[1].set_xlim(low - pad, high + pad)
    axes[1].set_ylim(low - pad, high + pad)
    axes[1].set_xlabel("Observed DFT formation energy (eV atom$^{-1}$)")
    axes[1].set_ylabel("LOO prediction (eV atom$^{-1}$)")
    axes[1].set_title("(b) DFT energy calibration")
    figure.tight_layout()
    figure.savefig(
        figures_dir / "Figure2_model_validation.pdf",
        bbox_inches="tight",
        metadata=PDF_METADATA,
    )
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharex=True)
    palette = {
        "group_gated_da_tpp": "#D62728",
        "group_only": "#E45756",
        "dft_evaluable_greedy": "#1F77B4",
        "predicted_target_greedy": "#4C78A8",
        "random_sampling": "#7F7F7F",
    }
    for method, group in trajectories.groupby("method", sort=False):
        color = palette.get(method, "0.65")
        width = 1.8 if method in palette else 0.9
        label = _display_method(method)
        x = group["checkpoint"].astype(float).to_numpy()
        ml_labeled = group["mean_ml_labeled_qualified"].astype(float).to_numpy()
        expected = group["mean_expected_qualified"].astype(float).to_numpy()
        axes[0].plot(x, ml_labeled, label=label, color=color, linewidth=width)
        axes[1].plot(x, expected, label=label, color=color, linewidth=width)
    axes[0].set_title("(a) ML-labeled qualified count")
    axes[1].set_title("(b) Expected qualified yield")
    for axis in axes:
        axis.set_xlabel("Query budget")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Candidate count")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    figure.suptitle(
        "Target-qualified ML-assisted replay (not observed DFT)"
    )
    figure.tight_layout()
    figure.savefig(
        figures_dir / "Figure3_expected_yield.pdf",
        bbox_inches="tight",
        metadata=PDF_METADATA,
    )
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(4.8, 3.1))
    for policy, group in observed_curve.groupby("policy", sort=False):
        axis.step(
            group["round"].astype(float),
            group["cumulative_DFT_evaluable"].astype(float),
            where="post",
            label=policy,
            linewidth=1.8,
        )
    axis.set_xlabel("Recorded acquisition round")
    axis.set_ylabel("Cumulative DFT-evaluable candidates")
    axis.set_title("Observed retrospective selected subset (12 candidates)")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(
        figures_dir / "Figure4_actual_dft_ordering.pdf",
        bbox_inches="tight",
        metadata=PDF_METADATA,
    )
    plt.close(figure)

    generated_paths = [
        figures_dir / "Figure1_multifidelity_workflow.pdf",
        figures_dir / "Figure2_model_validation.pdf",
        figures_dir / "Figure3_expected_yield.pdf",
        figures_dir / "Figure4_actual_dft_ordering.pdf",
        tables_dir / "multifidelity_metrics_macros.tex",
        tables_dir / "table_model_validation.tex",
        tables_dir / "table_expected_yield.tex",
        tables_dir / "table_actual_dft_ordering.tex",
        source_dir / "model_validation.csv",
        source_dir / "energy_calibration_best_oof.csv",
        source_dir / "multifidelity_expected_yield.csv",
        source_dir / "multifidelity_expected_yield_trajectories.csv",
        source_dir / "actual_dft_ordering.csv",
        source_dir / "actual_dft_curve.csv",
        source_dir / "selector_equivalence_audit.csv",
    ]
    manifest_rows = []
    for path in sorted(generated_paths):
        manifest_rows.append(
            {
                "relative_path": path.relative_to(output_dir).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    pd.DataFrame(manifest_rows).to_csv(
        source_dir / "manuscript_asset_sha256.csv",
        index=False,
    )
