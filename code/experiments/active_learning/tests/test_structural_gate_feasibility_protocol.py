import json

import pytest

from experiments.reproducibility.formal_protocol import (
    FormalProtocolError,
    load_formal_protocol,
)
from experiments.reproducibility import run_paired_dataset_job as runner
from active_learning_energy_gate_ablation import build_parser


def write_protocol(tmp_path):
    path = tmp_path / "protocol.json"
    path.write_text(
        json.dumps(
            {
                "protocol_version": "egdatpp_structgate_feas_v1",
                "phase": "structural_group_feasibility",
                "dataset": "limo",
                "allowed_seeds": [111, 112, 113, 114, 115],
                "allowed_methods": [
                    "structural_group_gate",
                    "structural_group_gate_q95",
                ],
                "mc_passes": 30,
                "M0": 0.75,
                "G0": 0.50,
                "alpha": 0.10,
                "beta": 0.20,
                "gamma": 0.05,
                "group_key_mode": "structure_matcher_cluster",
                "group_key_map_relative_path": (
                    "configs/structural_group_feasibility/structure_matcher_group_map.csv"
                ),
                "frozen": True,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_structural_feasibility_protocol_accepts_only_heldout_cohort(tmp_path):
    protocol = load_formal_protocol(write_protocol(tmp_path))
    base = runner.dataset_configs()["limo"]

    protocol.resolve_dataset_config(base, method="structural_group_gate", seed=111)
    protocol.resolve_dataset_config(base, method="structural_group_gate_q95", seed=115)

    with pytest.raises(FormalProtocolError, match="not allowed"):
        protocol.resolve_dataset_config(base, method="structural_group_gate", seed=110)


def test_selector_arguments_forward_new_protocol_version():
    arguments = runner.selector_protocol_arguments(
        experiment_seed=111,
        acquisition_round=1,
        model_refit_index=0,
        protocol_version="egdatpp_structgate_feas_v1",
    )

    assert arguments[-2:] == ["--protocol-version", "egdatpp_structgate_feas_v1"]

    with pytest.raises(ValueError, match="unsupported protocol version"):
        runner.selector_protocol_arguments(
            experiment_seed=111,
            acquisition_round=1,
            model_refit_index=0,
            protocol_version="made_up_version",
        )


def test_experiment_seed_parser_accepts_only_new_declared_cohort():
    assert runner.parse_experiment_seed("111") == 111
    assert runner.parse_experiment_seed("115") == 115
    with pytest.raises(Exception, match="declared cohort"):
        runner.parse_experiment_seed("116")


def test_selector_cli_accepts_new_protocol_version():
    protocol_action = next(
        action for action in build_parser()._actions if action.dest == "protocol_version"
    )
    assert "egdatpp_structgate_feas_v1" in protocol_action.choices


def test_runner_exposes_both_structural_gate_methods():
    assert runner.METHOD_SPECS["structural_group_gate"] == {
        "display_name": "Structural-Group Gate",
        "selection_method_name": "structural_group_gate",
        "ablation_mode": "full",
        "quality_safeguard_fraction": None,
    }
    assert runner.METHOD_SPECS["structural_group_gate_q95"] == {
        "display_name": "Structural-Group Gate + Q95",
        "selection_method_name": "structural_group_gate_q95",
        "ablation_mode": "full",
        "quality_safeguard_fraction": 0.95,
    }
