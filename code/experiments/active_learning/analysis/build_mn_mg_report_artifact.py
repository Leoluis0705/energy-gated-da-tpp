from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    clean = frame.where(pd.notna(frame), None)
    return clean.to_dict(orient="records")


def build_artifact(analysis_dir: Path) -> dict[str, object]:
    tasks = pd.read_csv(analysis_dir / "mn_mg_interval_tasks.csv")
    summary = pd.read_csv(analysis_dir / "mn_mg_multiseed_summary.csv")
    paired = pd.read_csv(analysis_dir / "mn_mg_gate_greedy_paired.csv")
    paired_stats = pd.read_csv(analysis_dir / "mn_mg_gate_greedy_statistics.csv")
    density = pd.read_csv(analysis_dir / "mn_mg_target_density.csv")
    hidden = pd.read_csv(analysis_dir / "mn_mg_hidden_evaluability_summary.csv")
    model_cv = pd.read_csv(
        analysis_dir.parent.parent
        / "gpu_mn_mg_cgcnn_20260803"
        / "experiments"
        / "hidden_evaluability"
        / "results"
        / "three_system_low_data_v3_joint_endpoint"
        / "dft_evaluability_model_cv.csv"
    )
    candidate_master = pd.read_csv(
        analysis_dir.parent.parent
        / "gpu_mn_mg_cgcnn_20260803"
        / "experiments"
        / "dft_audit"
        / "candidate_pool_master.csv"
    )

    summary_view = summary.loc[
        :,
        [
            "task_label",
            "method_label",
            "AUTC_mean",
            "AUTC_sd",
            "recovery_at_80_mean",
            "recovery_at_160_mean",
            "recovery_at_240_mean",
            "recovery_at_320_mean",
            "AUTC_rank",
        ],
    ].copy()
    paired_view = paired.loc[:, ["task", "seed", "Gate_minus_Greedy_AUTC"]].copy()
    paired_view["task_label"] = paired_view["task"].map(
        {"mn": "Mn-anchored", "mg": "Mg-anchored"}
    )
    density_view = density.loc[
        density["m_element"] != "ALL",
        ["task_label", "m_element", "target_count", "target_fraction_within_task"],
    ].copy()
    hidden_view = hidden.loc[
        hidden["method"].isin(
            ["energy_gated_da_tpp", "predicted_target_greedy", "gradient_norm_hybrid"]
        ),
        [
            "task_label",
            "method_label",
            "checkpoint",
            "expected_evaluable_target_count_mean",
            "expected_evaluable_target_count_sd",
            "score_coverage_mean",
        ],
    ].copy()

    mn_stats = paired_stats.loc[paired_stats["task"] == "mn"].iloc[0]
    mg_stats = paired_stats.loc[paired_stats["task"] == "mg"].iloc[0]
    mn_task = tasks.loc[tasks["task"] == "mn"].iloc[0]
    mg_task = tasks.loc[tasks["task"] == "mg"].iloc[0]
    best_model = model_cv.sort_values("loo_roc_auc", ascending=False).iloc[0]

    metrics = pd.DataFrame(
        [
            {
                "formal_runs": 180,
                "mn_gate_minus_greedy": float(mn_stats["mean_difference"]),
                "mg_gate_minus_greedy": float(mg_stats["mean_difference"]),
                "hidden_model_auc": float(best_model["loo_roc_auc"]),
            }
        ]
    )

    energy = candidate_master["alignn_formation_energy_eV_atom"].astype(float)
    element_mean = candidate_master.groupby("m_element")[
        "alignn_formation_energy_eV_atom"
    ].transform("mean")
    composition_r2 = 1.0 - float(((energy - element_mean) ** 2).sum()) / float(
        ((energy - energy.mean()) ** 2).sum()
    )
    target_rate_rows: list[dict[str, object]] = []
    for task_code, task_label, low, high in (
        ("mn", "Mn-anchored", -2.1, -1.9),
        ("mg", "Mg-anchored", -2.3, -2.1),
    ):
        target = energy.between(low, high, inclusive="both")
        by_element = (
            candidate_master.assign(target=target)
            .groupby("m_element", as_index=False)
            .agg(pool_n=("candidate_id", "size"), target_n=("target", "sum"))
        )
        by_element["within_element_target_rate"] = (
            by_element["target_n"] / by_element["pool_n"]
        )
        by_element["share_of_task_targets"] = (
            by_element["target_n"] / by_element["target_n"].sum()
        )
        by_element["task"] = task_code
        by_element["task_label"] = task_label
        target_rate_rows.extend(_records(by_element))
    target_rate = pd.DataFrame(target_rate_rows)

    diagnostic_rows: list[dict[str, object]] = []
    for task_code, task_label in (("mn", "Mn-anchored"), ("mg", "Mg-anchored")):
        gate = summary.loc[
            (summary["task"] == task_code)
            & (summary["method"] == "energy_gated_da_tpp")
        ].iloc[0]
        greedy = summary.loc[
            (summary["task"] == task_code)
            & (summary["method"] == "predicted_target_greedy")
        ].iloc[0]
        run_rows = pd.read_csv(analysis_dir / "mn_mg_multiseed_results.csv")
        gate_runs = run_rows.loc[
            (run_rows["task"] == task_code)
            & (run_rows["method"] == "energy_gated_da_tpp")
        ]
        diagnostic_rows.append(
            {
                "task_label": task_label,
                "correction_round_fraction": float(
                    gate_runs["correction_rounds"].mean()
                    / (
                        gate_runs["correction_rounds"].mean()
                        + gate_runs["direct_rounds"].mean()
                    )
                ),
                "mean_replaced_positions": float(
                    gate_runs["total_correction_replacements"].mean()
                ),
                "mean_net_target_gain_from_correction": float(
                    gate_runs["total_correction_target_gain"].mean()
                ),
                "gate_minus_greedy_recovery_at_80": float(
                    gate["recovery_at_80_mean"] - greedy["recovery_at_80_mean"]
                ),
                "gate_minus_greedy_all_clusters_at_80": float(
                    gate_runs["unique_structure_clusters_at_80"].mean()
                    - run_rows.loc[
                        (run_rows["task"] == task_code)
                        & (run_rows["method"] == "predicted_target_greedy"),
                        "unique_structure_clusters_at_80",
                    ].mean()
                ),
                "gate_minus_greedy_target_clusters_at_80": float(
                    gate_runs["unique_target_structure_clusters_at_80"].mean()
                    - run_rows.loc[
                        (run_rows["task"] == task_code)
                        & (run_rows["method"] == "predicted_target_greedy"),
                        "unique_target_structure_clusters_at_80",
                    ].mean()
                ),
            }
        )
    diagnostic_summary = pd.DataFrame(diagnostic_rows)

    generated = datetime.now(timezone.utc).isoformat()
    sources = [
        {
            "id": "formal_results",
            "label": "Independently recomputed 180-run formal results",
            "path": "mn_mg_multiseed_results.csv",
            "query": {
                "description": "AUTC, checkpoint recovery, routing, and seed-level metrics recomputed from raw histories.",
                "language": "Python",
                "sql": (
                    "SELECT task_label, method_label, AVG(AUTC) AS AUTC_mean, "
                    "STDDEV_SAMP(AUTC) AS AUTC_sd, AVG(recovery_at_80) AS recovery_at_80_mean, "
                    "AVG(recovery_at_160) AS recovery_at_160_mean, "
                    "AVG(recovery_at_240) AS recovery_at_240_mean, "
                    "AVG(recovery_at_320) AS recovery_at_320_mean "
                    "FROM read_csv_auto('mn_mg_multiseed_results.csv') "
                    "GROUP BY task_label, method_label;"
                ),
                "tables_used": ["mn_mg_multiseed_results.csv"],
                "filters": ["tasks=mn,mg", "methods=9", "seeds=101-110", "budget=320"],
                "metric_definitions": [
                    "Normalized AUTC uses a left-continuous cumulative-recovery curve divided by target_count times budget.",
                    "Gate-minus-Greedy is paired within task and initial-set seed.",
                ],
            },
        },
        {
            "id": "task_inventory",
            "label": "Frozen proxy-interval task inventory",
            "path": "mn_mg_interval_tasks.csv",
            "query": {
                "description": "Target intervals, density, elemental composition, structural clusters, and DFT/proxy anchor separation.",
                "language": "Python",
                "sql": (
                    "SELECT task_label, m_element, target_count, target_fraction_within_task "
                    "FROM read_csv_auto('mn_mg_target_density.csv') WHERE m_element <> 'ALL';"
                ),
                "tables_used": ["mn_mg_interval_tasks.csv", "mn_mg_target_density.csv"],
            },
        },
        {
            "id": "hidden_audit",
            "label": "Post-selection hidden DFT-evaluability audit",
            "path": "mn_mg_hidden_evaluability_audit.csv",
            "query": {
                "description": "Expected DFT-evaluable target count computed after selection and never used by acquisition.",
                "language": "Python",
                "sql": (
                    "SELECT task_label, method_label, checkpoint, "
                    "AVG(expected_evaluable_target_count) AS expected_evaluable_target_count_mean, "
                    "STDDEV_SAMP(expected_evaluable_target_count) AS expected_evaluable_target_count_sd, "
                    "AVG(score_coverage) AS score_coverage_mean "
                    "FROM read_csv_auto('mn_mg_hidden_evaluability_audit.csv') "
                    "GROUP BY task_label, method_label, checkpoint;"
                ),
                "tables_used": ["mn_mg_hidden_evaluability_audit.csv"],
                "metric_definitions": [
                    "Expected count is the sum of p(DFT evaluable) over scored selected target candidates.",
                    "This is a model expectation, not an observed DFT success count.",
                ],
            },
        },
        {
            "id": "pool_algorithm_diagnosis",
            "label": "Pool composition and correction-route diagnosis",
            "path": "candidate_pool_master.csv and mn_mg_multiseed_results.csv",
            "query": {
                "description": "Element-conditioned target prevalence and the target/diversity tradeoff introduced by the correction route.",
                "language": "Python",
                "sql": (
                    "SELECT m_element, COUNT(*) AS pool_n, "
                    "SUM(alignn_formation_energy_eV_atom BETWEEN target_low AND target_high) AS target_n "
                    "FROM candidate_pool_master GROUP BY m_element; "
                    "SELECT task, AVG(correction_rounds), AVG(total_correction_replacements), "
                    "AVG(total_correction_target_gain) FROM mn_mg_multiseed_results "
                    "WHERE method='energy_gated_da_tpp' GROUP BY task;"
                ),
                "tables_used": [
                    "candidate_pool_master.csv",
                    "mn_mg_multiseed_results.csv",
                ],
                "metric_definitions": [
                    "Composition R-squared is the variance in ALIGNN proxy formation energy explained by the seven M-element group means.",
                    "Correction target gain compares oracle target membership of the selected correction batch with the direct top-probability batch at each correction round.",
                ],
            },
        },
    ]

    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Mn/Mg-Anchored Interval Active-Learning Audit",
            "description": "Independent recomputation of 180 GPU runs, task-scope validation, and manuscript claim boundaries.",
            "generatedAt": generated,
            "sources": sources,
            "cards": [
                {
                    "id": "headline_metrics",
                    "dataset": "headline_metrics",
                    "sourceId": "formal_results",
                    "metrics": [
                        {"label": "Formal runs", "field": "formal_runs", "format": "number"},
                        {"label": "Mn Gate - Greedy AUTC", "field": "mn_gate_minus_greedy", "format": "number", "signed": True},
                        {"label": "Mg-anchor Gate - Greedy AUTC", "field": "mg_gate_minus_greedy", "format": "number", "signed": True},
                        {"label": "Hidden-model LOO ROC-AUC", "field": "hidden_model_auc", "format": "number"},
                    ],
                }
            ],
            "charts": [
                {
                    "id": "autc_chart",
                    "title": "Normalized AUTC by method",
                    "subtitle": "Mean across ten independent initial-set seeds.",
                    "intent": "comparison",
                    "type": "horizontalBar",
                    "dataset": "method_summary",
                    "sourceId": "formal_results",
                    "encodings": {
                        "x": {"field": "method_label", "type": "nominal", "label": "Method"},
                        "y": {"field": "AUTC_mean", "type": "quantitative", "label": "Normalized AUTC"},
                        "color": {"field": "task_label", "type": "nominal", "label": "Task"},
                    },
                    "maxRows": 18,
                },
                {
                    "id": "paired_delta_chart",
                    "title": "Seed-level Full Gate versus Greedy AUTC",
                    "subtitle": "Positive values favor Full Gate.",
                    "intent": "distribution",
                    "type": "scatter",
                    "dataset": "paired_seed",
                    "sourceId": "formal_results",
                    "encodings": {
                        "x": {"field": "seed", "type": "ordinal", "label": "Initial-set seed"},
                        "y": {"field": "Gate_minus_Greedy_AUTC", "type": "quantitative", "label": "Gate - Greedy AUTC"},
                        "color": {"field": "task_label", "type": "nominal", "label": "Task"},
                    },
                    "referenceLines": [
                        {"axis": "y", "value": 0, "label": "No difference", "lineStyle": "dashed"}
                    ],
                },
                {
                    "id": "composition_chart",
                    "title": "Actual target composition of the proxy intervals",
                    "subtitle": "Both tasks were run on the complete frozen 640-candidate pool.",
                    "intent": "composition",
                    "type": "stackedBar",
                    "dataset": "target_composition",
                    "sourceId": "task_inventory",
                    "encodings": {
                        "x": {"field": "task_label", "type": "nominal", "label": "Task"},
                        "y": {"field": "target_count", "type": "quantitative", "aggregate": "sum", "label": "Target candidates"},
                        "color": {"field": "m_element", "type": "nominal", "label": "M element"},
                    },
                },
                {
                    "id": "element_target_rate_chart",
                    "title": "Target prevalence within each elemental subgroup",
                    "subtitle": "The interval labels almost identify the M element directly.",
                    "intent": "comparison",
                    "type": "bar",
                    "dataset": "element_target_rate",
                    "sourceId": "pool_algorithm_diagnosis",
                    "encodings": {
                        "x": {"field": "m_element", "type": "nominal", "label": "M element"},
                        "y": {"field": "within_element_target_rate", "type": "quantitative", "label": "Target prevalence"},
                        "color": {"field": "task_label", "type": "nominal", "label": "Task"},
                    },
                },
            ],
            "tables": [
                {
                    "id": "method_table",
                    "title": "All-method summary",
                    "dataset": "method_summary",
                    "sourceId": "formal_results",
                    "defaultSort": {"field": "AUTC_mean", "direction": "desc"},
                    "columns": [
                        {"field": "task_label", "label": "Task"},
                        {"field": "method_label", "label": "Method"},
                        {"field": "AUTC_mean", "label": "Mean AUTC", "format": "number"},
                        {"field": "AUTC_sd", "label": "AUTC SD", "format": "number"},
                        {"field": "recovery_at_80_mean", "label": "R@80", "format": "number"},
                        {"field": "recovery_at_160_mean", "label": "R@160", "format": "number"},
                        {"field": "recovery_at_240_mean", "label": "R@240", "format": "number"},
                        {"field": "recovery_at_320_mean", "label": "R@320", "format": "number"},
                    ],
                },
                {
                    "id": "hidden_table",
                    "title": "Hidden DFT-evaluability audit",
                    "subtitle": "Model expectations, not observed new DFT outcomes.",
                    "dataset": "hidden_summary",
                    "sourceId": "hidden_audit",
                    "defaultSort": {"field": "checkpoint", "direction": "asc"},
                    "columns": [
                        {"field": "task_label", "label": "Task"},
                        {"field": "method_label", "label": "Method"},
                        {"field": "checkpoint", "label": "Queries", "format": "number"},
                        {"field": "expected_evaluable_target_count_mean", "label": "Expected evaluable targets", "format": "number"},
                        {"field": "expected_evaluable_target_count_sd", "label": "SD", "format": "number"},
                        {"field": "score_coverage_mean", "label": "Score coverage", "format": "percent"},
                    ],
                },
                {
                    "id": "diagnostic_table",
                    "title": "Correction-route cost and diversity tradeoff",
                    "subtitle": "Mean across ten initial-set seeds at the 320-query protocol.",
                    "dataset": "diagnostic_summary",
                    "sourceId": "pool_algorithm_diagnosis",
                    "columns": [
                        {"field": "task_label", "label": "Task"},
                        {"field": "correction_round_fraction", "label": "Correction-round share", "format": "percent"},
                        {"field": "mean_replaced_positions", "label": "Replaced positions", "format": "number"},
                        {"field": "mean_net_target_gain_from_correction", "label": "Net target gain", "format": "number", "movement": True},
                        {"field": "gate_minus_greedy_recovery_at_80", "label": "Gate-Greedy R@80", "format": "number", "movement": True},
                        {"field": "gate_minus_greedy_all_clusters_at_80", "label": "All-cluster delta", "format": "number", "movement": True},
                        {"field": "gate_minus_greedy_target_clusters_at_80", "label": "Target-cluster delta", "format": "number", "movement": True},
                    ],
                },
            ],
            "blocks": [
                {"id": "title", "type": "markdown", "layout": "full", "body": "# Mn/Mg-Anchored Interval Active-Learning Audit"},
                {
                    "id": "summary",
                    "type": "markdown",
                    "layout": "full",
                    "sourceId": "formal_results",
                    "body": (
                        "## Headline result\n\n"
                        f"On the Mn-anchored task, Full Gate minus Greedy AUTC was {mn_stats['mean_difference']:+.4f}; Full Gate lost in all ten paired seeds. "
                        f"On the Mg-anchored task, the mean difference was {mg_stats['mean_difference']:+.4f}, but the result was unstable. "
                        f"Moreover, {mg_task['dominant_element_fraction']:.1%} of that target set was Cr, so it cannot establish Mg-specific robustness."
                    ),
                },
                {"id": "metrics", "type": "metric-strip", "layout": "full", "cardIds": ["headline_metrics"]},
                {"id": "autc", "type": "chart", "layout": "full", "chartId": "autc_chart"},
                {"id": "delta", "type": "chart", "layout": "half", "chartId": "paired_delta_chart"},
                {"id": "composition", "type": "chart", "layout": "half", "chartId": "composition_chart"},
                {
                    "id": "task_scope",
                    "type": "markdown",
                    "layout": "full",
                    "sourceId": "task_inventory",
                    "body": (
                        "## Scope validation\n\n"
                        f"The Mn interval contained {int(mn_task['target_count'])} targets, of which {mn_task['dominant_element_fraction']:.1%} were Mn. "
                        f"The Mg-anchored interval contained {int(mg_task['target_count'])} targets, of which {mg_task['dominant_element_fraction']:.1%} were Cr. "
                        "Both tasks used the complete 640-candidate pool rather than element-only subpools. Only the 0.2 eV/atom interval width was run."
                    ),
                },
                {
                    "id": "pool_diagnosis",
                    "type": "markdown",
                    "layout": "full",
                    "sourceId": "pool_algorithm_diagnosis",
                    "body": (
                        "## The failure is a pool/group-key mismatch, not a failed GPU run\n\n"
                        f"The M element alone explains {composition_r2:.1%} of the variance in the frozen ALIGNN proxy labels. "
                        "Within the Mn subgroup, 120 of 127 candidates are targets for the Mn-anchored interval; within Cr, 112 of 114 candidates are targets for the Mg-anchored interval. "
                        "The selector's element-system group key therefore treats the very chemistry that defines target membership as redundancy and pushes some high-probability targets out of the batch. "
                        "The chart shows that these tasks are effectively elemental classification problems before they are structural-ranking problems."
                    ),
                },
                {"id": "element_rates", "type": "chart", "layout": "full", "chartId": "element_target_rate_chart"},
                {
                    "id": "correction_diagnosis",
                    "type": "markdown",
                    "layout": "full",
                    "sourceId": "pool_algorithm_diagnosis",
                    "body": (
                        "## Correction increases broad diversity but loses target candidates\n\n"
                        "Full Gate entered the correction route in 93.7% of Mn rounds and 90.5% of Mg-anchored rounds. "
                        "At 80 queries it gained six total structural clusters on Mn but only 0.5 target clusters while losing 9.5 recovered targets relative to Greedy; the same pattern appears in the Mg-anchored task. "
                        "The correction route is therefore functioning as coded, but its diversity objective is misaligned with these interval-defined target sets."
                    ),
                },
                {"id": "correction_table_block", "type": "table", "layout": "full", "tableId": "diagnostic_table"},
                {"id": "methods", "type": "table", "layout": "full", "tableId": "method_table"},
                {
                    "id": "hidden_note",
                    "type": "markdown",
                    "layout": "full",
                    "sourceId": "hidden_audit",
                    "body": (
                        "## Hidden DFT-evaluability audit\n\n"
                        f"The best model used {int(best_model['n'])} historical DFT attempts and achieved LOO ROC-AUC={best_model['loo_roc_auc']:.2f}. "
                        "It was applied only after acquisition. Expected counts must not be reported as observed DFT success counts."
                    ),
                },
                {"id": "hidden", "type": "table", "layout": "full", "tableId": "hidden_table"},
                {
                    "id": "claims",
                    "type": "markdown",
                    "layout": "full",
                    "body": (
                        "## Manuscript claim boundary\n\n"
                        "The manuscript may report a negative robustness result on the Mn task and the exact behavioral equivalence Full Gate = Group-only and Margin-only = Greedy. "
                        "It must not claim cross-element superiority over Greedy, Mg-system validation, interval-width robustness, or improved observed DFT success."
                    ),
                },
                {
                    "id": "rescue_plan",
                    "type": "markdown",
                    "layout": "full",
                    "body": (
                        "## Fastest defensible rescue\n\n"
                        "Do not tune the interval or remove Greedy. First replace the element-system group key with a label-blind structural-cluster key. Test a separate ablation that adds a quality-loss safeguard rejecting a diversity batch when its summed interval-hit probability falls materially below the direct batch. "
                        "Run a five-seed diagnostic with Greedy, the retained Gate, the structural-group Gate, structural-group plus safeguard, and Gradient-norm hybrid. Freeze the group map and safeguard on development seeds before any ten-seed confirmation. "
                        "If the structural version merely matches Greedy, report an applicability boundary; if it improves both target recovery and target-cluster coverage, proceed to the formal run."
                    ),
                },
                {
                    "id": "further_questions",
                    "type": "markdown",
                    "layout": "full",
                    "body": (
                        "## Questions that remain open\n\n"
                        "The immediate diagnostic should answer whether structural grouping removes the observed target loss, whether the quality safeguard is necessary after grouping is corrected, and whether any gain survives on unseen initial-set seeds. "
                        "No revised method should be described as preregistered or independent of these observed negative results."
                    ),
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated,
            "status": "ready",
            "datasets": {
                "headline_metrics": _records(metrics),
                "method_summary": _records(summary_view),
                "paired_seed": _records(paired_view),
                "target_composition": _records(density_view),
                "hidden_summary": _records(hidden_view),
                "element_target_rate": _records(target_rate),
                "diagnostic_summary": _records(diagnostic_summary),
            },
            "accessIssues": [],
        },
        "sources": sources,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("analysis_dir", type=Path)
    args = parser.parse_args()
    artifact = build_artifact(args.analysis_dir)
    (args.analysis_dir / "artifact.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
