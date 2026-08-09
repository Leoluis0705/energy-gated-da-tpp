import argparse

import pytest

from experiments.reproducibility import run_paired_dataset_job as runner


@pytest.mark.parametrize("value", ["0", "4", "5", "14", "15", "24", "25", "29"])
def test_parse_experiment_seed_accepts_declared_cohorts(value):
    assert runner.parse_experiment_seed(value) == int(value)


@pytest.mark.parametrize("value", ["-1", "30"])
def test_parse_experiment_seed_rejects_outside_declared_cohorts(value):
    with pytest.raises(argparse.ArgumentTypeError, match="0..29"):
        runner.parse_experiment_seed(value)


def test_formal_method_map_covers_all_five_route_rules():
    assert runner.METHOD_SPECS["interval_hit_greedy"]["ablation_mode"] == "p_hit_greedy"
    assert runner.METHOD_SPECS["always_da_tpp"]["ablation_mode"] == "always_diversity"
    assert runner.METHOD_SPECS["margin_only_gate"]["ablation_mode"] == "gate_no_concentration"
    assert runner.METHOD_SPECS["group_only_gate"]["ablation_mode"] == "gate_no_margin"
    assert runner.METHOD_SPECS["energy_gated_da_tpp"]["ablation_mode"] == "full"
