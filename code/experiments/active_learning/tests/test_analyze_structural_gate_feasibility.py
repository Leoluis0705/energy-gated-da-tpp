import math

import pandas as pd

from analysis.analyze_structural_gate_feasibility import (
    _build_chinese_report,
    _build_claim_boundary,
    _checkpoint_metrics,
    _plot_outputs,
    _trace_column_sum,
    build_method_summary,
    build_paired,
    classify_decision,
)


REVISED = ("structural_group_gate", "structural_group_gate_q95")


def decision_rows(
    *,
    autc_delta,
    recovery_delta,
    cluster_delta,
    correction_gain,
    worst_seed_delta=None,
):
    rows = []
    for method in REVISED:
        for task in ("mn", "mg"):
            for seed in range(111, 116):
                delta = float(autc_delta)
                if worst_seed_delta is not None and seed == 115:
                    delta = float(worst_seed_delta)
                rows.append(
                    {
                        "method": method,
                        "task": task,
                        "seed": seed,
                        "autc_delta": delta,
                        "recovery_at_80_delta": float(recovery_delta),
                        "target_cluster_at_80_delta": float(cluster_delta),
                        "cumulative_correction_target_gain": float(correction_gain),
                    }
                )
    return pd.DataFrame(rows)


def test_strong_go_requires_nonnegative_recovery_coverage_and_gain_on_both_tasks():
    rows = decision_rows(
        autc_delta=0.01,
        recovery_delta=0,
        cluster_delta=0,
        correction_gain=0,
    )

    assert classify_decision(rows) == "STRONG_GO"


def test_conditional_go_supports_tradeoff_but_not_superiority():
    rows = decision_rows(
        autc_delta=-0.003,
        recovery_delta=-1,
        cluster_delta=1,
        correction_gain=-0.2,
        worst_seed_delta=-0.01,
    )

    assert classify_decision(rows) == "CONDITIONAL_GO"


def test_stop_when_both_revised_gates_miss_frozen_go_rules():
    rows = decision_rows(
        autc_delta=-0.02,
        recovery_delta=-3,
        cluster_delta=0,
        correction_gain=-4,
    )

    assert classify_decision(rows) == "STOP"


def test_decision_rejects_incomplete_heldout_grid():
    rows = decision_rows(
        autc_delta=0.01,
        recovery_delta=0,
        cluster_delta=0,
        correction_gain=0,
    ).query("seed != 115")

    try:
        classify_decision(rows)
    except ValueError as error:
        assert "complete five-seed" in str(error)
    else:
        raise AssertionError("incomplete held-out grid was accepted")


def test_report_templates_preserve_chinese_and_heldout_seed_range():
    summary = pd.DataFrame(
        [{"task": "mn", "method": "structural_group_gate", "mean_autc_delta": 0.01}]
    )

    report = _build_chinese_report("STRONG_GO", summary, summary)
    boundary = _build_claim_boundary("STRONG_GO")

    assert "结构分组 Gate 可行性实验报告" in report
    assert "仅为 post-selection 预测审计，不是真实 DFT 结果" in report
    assert "方法汇总" in report
    assert "相对 Greedy 的配对差值" in report
    assert "held-out seeds 111–115" in boundary


def test_paired_comparison_rejects_mismatched_initial_sets():
    common = {
        "task": "mn",
        "seed": 111,
        "autc": 0.5,
        "recovery_at_80": 10,
        "target_cluster_at_80": 4,
        "hidden_expected_evaluable_targets_at_80": 6.0,
        "cumulative_correction_target_gain": 0.0,
    }
    frame = pd.DataFrame(
        [
            {**common, "method": "predicted_target_greedy", "initial_set_sha256": "aaa"},
            {**common, "method": "structural_group_gate", "initial_set_sha256": "bbb"},
        ]
    )

    try:
        build_paired(frame)
    except ValueError as error:
        assert "initial-set hash" in str(error)
    else:
        raise AssertionError("mismatched initial sets were accepted")


def test_plot_outputs_create_all_frozen_figure_artifacts(tmp_path):
    per_seed_rows = []
    paired_rows = []
    methods = (
        "predicted_target_greedy",
        "energy_gated_da_tpp",
        "structural_group_gate",
        "structural_group_gate_q95",
        "gradient_norm_hybrid",
    )
    for task in ("mn", "mg"):
        for method_index, method in enumerate(methods):
            for seed in range(111, 116):
                row = {"task": task, "method": method, "seed": seed}
                for checkpoint in (80, 160, 240, 320):
                    row[f"recovery_at_{checkpoint}"] = checkpoint / 8 + method_index
                    row[f"target_cluster_at_{checkpoint}"] = checkpoint / 40 + method_index
                per_seed_rows.append(row)
                if method != "predicted_target_greedy":
                    paired_rows.append(
                        {
                            "task": task,
                            "method": method,
                            "seed": seed,
                            "autc_delta": 0.001 * method_index,
                            "cumulative_correction_target_gain": method_index - 1,
                        }
                    )

    _plot_outputs(pd.DataFrame(per_seed_rows), pd.DataFrame(paired_rows), tmp_path)

    for stem in (
        "figure_recovery",
        "figure_autc",
        "figure_target_cluster_coverage",
        "figure_correction_loss",
    ):
        assert (tmp_path / f"{stem}.pdf").is_file()
        assert (tmp_path / f"{stem}.png").is_file()


def test_trace_column_sum_preserves_selected_p_hit_and_missingness():
    trace = pd.DataFrame({"selected_batch_p_hit_sum": [1.0, None, 2.5]})

    assert _trace_column_sum(trace, "selected_batch_p_hit_sum") == 3.5
    assert math.isnan(_trace_column_sum(trace, "not_recorded"))


def test_hidden_evaluability_requires_scores_only_for_recovered_targets():
    history = pd.DataFrame(
        [
            {"id": "unscored_non_target", "target_label": 0},
            {"id": "scored_target", "target_label": 1},
        ]
    )
    group_map = {"unscored_non_target": "g0", "scored_target": "g1"}

    metrics = _checkpoint_metrics(
        history,
        checkpoint=2,
        group_map=group_map,
        evaluability={"scored_target": 0.8},
    )

    assert metrics["recovery_at_2"] == 1
    assert metrics["hidden_expected_evaluable_targets_at_2"] == 0.8


def test_paired_comparison_includes_all_frozen_checkpoints():
    rows = []
    for method, offset in (("predicted_target_greedy", 0), ("structural_group_gate", 2)):
        row = {
            "task": "mn",
            "seed": 111,
            "method": method,
            "initial_set_sha256": "same",
            "autc": 0.5 + offset / 100,
            "cumulative_correction_target_gain": float(offset),
        }
        for checkpoint in (80, 160, 240, 320):
            row[f"recovery_at_{checkpoint}"] = checkpoint // 8 + offset
            row[f"recovery_rate_at_{checkpoint}"] = checkpoint / 640 + offset / 100
            row[f"queries_per_target_at_{checkpoint}"] = checkpoint / (checkpoint // 8 + offset)
            row[f"target_cluster_at_{checkpoint}"] = checkpoint // 40 + offset
            row[f"hidden_expected_evaluable_targets_at_{checkpoint}"] = checkpoint / 10 + offset
        rows.append(row)

    paired = build_paired(pd.DataFrame(rows)).iloc[0]

    assert paired["recovery_at_320_delta"] == 2
    assert math.isclose(paired["recovery_rate_at_240_delta"], 0.02)
    assert paired["target_cluster_at_160_delta"] == 2
    assert paired["hidden_expected_evaluable_at_80_delta"] == 2


def test_method_summary_reports_every_method_and_checkpoint():
    rows = []
    for seed, autc in ((111, 0.5), (112, 0.7)):
        row = {
            "task": "mn",
            "method": "predicted_target_greedy",
            "seed": seed,
            "autc": autc,
            "final_recovery": 100,
        }
        for checkpoint in (80, 160, 240, 320):
            row[f"recovery_at_{checkpoint}"] = checkpoint // 8
            row[f"recovery_rate_at_{checkpoint}"] = checkpoint / 640
            row[f"queries_per_target_at_{checkpoint}"] = 8.0
            row[f"target_cluster_at_{checkpoint}"] = checkpoint // 40
            row[f"hidden_expected_evaluable_targets_at_{checkpoint}"] = checkpoint / 10
        rows.append(row)

    summary = build_method_summary(pd.DataFrame(rows)).iloc[0]

    assert summary["mean_autc"] == 0.6
    assert summary["mean_recovery_at_320"] == 40
    assert summary["mean_hidden_evaluable_at_80"] == 8
