from __future__ import annotations

import inspect
from pathlib import Path

from analysis.run_gate_ablation_smoke import load_formal_gate_module
from experiments.reproducibility.run_paired_dataset_job import selector_protocol_arguments
from mc_dropout_seed_policy import (
    deterministic_mask_seed,
    mask_sequence_sha256,
    mc_mask_seeds,
)


def test_paired_methods_receive_the_same_mask_hash() -> None:
    gate_hash = mask_sequence_sha256(
        10,
        experiment_seed=3,
        acquisition_round=7,
        model_refit_index=6,
    )
    greedy_hash = mask_sequence_sha256(
        10,
        experiment_seed=3,
        acquisition_round=7,
        model_refit_index=6,
    )

    assert gate_hash == greedy_hash
    assert "method" not in inspect.signature(deterministic_mask_seed).parameters
    assert "method" not in inspect.signature(mc_mask_seeds).parameters


def test_different_experiment_seeds_have_different_mask_hashes() -> None:
    first = mask_sequence_sha256(
        30,
        experiment_seed=0,
        acquisition_round=1,
        model_refit_index=0,
    )
    second = mask_sequence_sha256(
        30,
        experiment_seed=1,
        acquisition_round=1,
        model_refit_index=0,
    )

    assert first != second


def test_repeated_runs_are_identical_and_passes_are_distinct() -> None:
    first = mc_mask_seeds(
        30,
        experiment_seed=4,
        acquisition_round=12,
        model_refit_index=11,
    )
    second = mc_mask_seeds(
        30,
        experiment_seed=4,
        acquisition_round=12,
        model_refit_index=11,
    )

    assert first == second
    assert len(first) == len(set(first)) == 30
    assert all(0 <= value < 2**63 for value in first)


def test_k_3_10_30_are_reproducible_nested_prefixes() -> None:
    kwargs = {
        "experiment_seed": 2,
        "acquisition_round": 5,
        "model_refit_index": 4,
    }
    seeds3 = mc_mask_seeds(3, **kwargs)
    seeds10 = mc_mask_seeds(10, **kwargs)
    seeds30 = mc_mask_seeds(30, **kwargs)

    assert seeds10[:3] == seeds3
    assert seeds30[:10] == seeds10
    assert mc_mask_seeds(3, **kwargs) == seeds3
    assert mc_mask_seeds(10, **kwargs) == seeds10
    assert mc_mask_seeds(30, **kwargs) == seeds30


def test_round_refit_and_pass_index_all_enter_the_hash() -> None:
    baseline = deterministic_mask_seed(3, 7, 0, 6)

    assert deterministic_mask_seed(3, 8, 0, 6) != baseline
    assert deterministic_mask_seed(3, 7, 1, 6) != baseline
    assert deterministic_mask_seed(3, 7, 0, 7) != baseline


def test_invalid_seed_components_and_k_are_rejected() -> None:
    import pytest

    with pytest.raises(ValueError, match="K"):
        mc_mask_seeds(0, experiment_seed=0, acquisition_round=1, model_refit_index=0)
    with pytest.raises(ValueError, match="non-negative"):
        deterministic_mask_seed(-1, 1, 0, 0)


def test_corrected_selector_cli_requires_experiment_and_refit_seed_state() -> None:
    archive = Path(__file__).resolve().parents[1]
    formal = load_formal_gate_module(
        archive / "active_learning_energy_gate_ablation.py",
        archive / "experiments/reproducibility/staging/paired_confirmation_server_20260712",
    )

    parser = formal.build_parser()
    actions = parser._option_string_actions
    assert actions["--experiment-seed"].required
    assert actions["--model-refit-index"].required
    assert "--method-name" not in actions


def test_runner_passes_method_independent_seed_state_to_selector() -> None:
    arguments = selector_protocol_arguments(
        experiment_seed=3,
        acquisition_round=7,
        model_refit_index=6,
    )

    assert arguments == [
        "--experiment-seed",
        "3",
        "--model-refit-index",
        "6",
        "--protocol-version",
        "egdatpp_psfix_v1",
    ]
    assert "method" not in inspect.signature(selector_protocol_arguments).parameters
