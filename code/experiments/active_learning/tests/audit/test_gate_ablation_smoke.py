from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import pandas as pd

from analysis.run_gate_ablation_smoke import (
    METHOD_ORDER,
    LIMO_FORMAL_PARAMS,
    _diversity_select,
    _restore_cached_runtime,
    _restore_cached_boolean,
    evaluate_gate_methods,
    load_formal_gate_module,
    route_for_method,
)


def test_forced_routes_are_invariant() -> None:
    for margin in (0.0, 2.0):
        for concentration in (0.1, 0.9):
            assert route_for_method("Interval-Hit Greedy", margin, concentration, 1.0, 0.75) == "direct"
            assert route_for_method("Always-DA-TPP", margin, concentration, 1.0, 0.75) == "correction"


def test_margin_only_ignores_group_condition() -> None:
    assert route_for_method("Margin-only Gate", 1.1, 0.1, 1.0, 0.75) == "direct"
    assert route_for_method("Margin-only Gate", 1.1, 0.99, 1.0, 0.75) == "direct"
    assert route_for_method("Margin-only Gate", 0.9, 0.1, 1.0, 0.75) == "correction"
    assert route_for_method("Margin-only Gate", 0.9, 0.99, 1.0, 0.75) == "correction"


def test_group_only_ignores_margin_condition() -> None:
    assert route_for_method("Group-only Gate", 0.1, 0.70, 1.0, 0.75) == "direct"
    assert route_for_method("Group-only Gate", 9.0, 0.70, 1.0, 0.75) == "direct"
    assert route_for_method("Group-only Gate", 0.1, 0.80, 1.0, 0.75) == "correction"
    assert route_for_method("Group-only Gate", 9.0, 0.80, 1.0, 0.75) == "correction"


def test_full_gate_matches_retained_formal_implementation_truth_table() -> None:
    archive = Path(__file__).resolve().parents[2]
    formal = load_formal_gate_module(
        archive / "active_learning_energy_gate_ablation.py",
        archive / "experiments/reproducibility/staging/paired_confirmation_server_20260712",
    )
    for margin in (0.0, 1.0, 1.2):
        for concentration in (0.5, 0.75, 0.9):
            expected = formal.gate_mode("full", margin, concentration, 1.0, 0.75)
            observed = route_for_method(
                "Full Energy-Gated DA-TPP", margin, concentration, 1.0, 0.75
            )
            assert observed == {
                "threshold_greedy": "direct",
                "diversity_aware": "correction",
            }[expected]


def test_correction_selector_matches_retained_formal_implementation() -> None:
    archive = Path(__file__).resolve().parents[2]
    formal = load_formal_gate_module(
        archive / "active_learning_energy_gate_ablation.py",
        archive / "experiments/reproducibility/staging/paired_confirmation_server_20260712",
    )
    ids = [f"candidate_{index}" for index in range(12)]
    p_hit = np.linspace(0.95, 0.40, 12)
    uncertainty = np.linspace(0.05, 0.95, 12)
    groups = ["A", "A", "A", "B", "B", "C", "D", "D", "E", "F", "G", "H"]
    features = np.arange(48, dtype=float).reshape(12, 4) + np.eye(12, 4)
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    observed = _diversity_select(
        ids, p_hit, uncertainty, groups, features, 5, 2, 0.10, 0.20, 0.10
    )
    expected = formal.diversity_select(
        ids, p_hit, uncertainty, groups, features, 5, 2, 0.10, 0.20, 0.10
    )[0]
    assert observed == expected


def test_evaluate_gate_methods_uses_one_shared_score_state() -> None:
    ids = [f"candidate_{index}" for index in range(8)]
    pseudo = np.array([-2.10, -2.11, -2.09, -2.12, -2.08, -2.13, -2.07, -2.14])
    raw_sigma = np.full(8, 0.05)
    groups = ["A", "A", "A", "A", "B", "C", "D", "E"]
    similarity = np.eye(8)
    result = evaluate_gate_methods(
        ids=ids,
        pseudo=pseudo,
        raw_sigma=raw_sigma,
        groups=groups,
        similarity=similarity,
        target_low=-2.18,
        target_high=-2.02,
        batch_size=3,
        margin_threshold=1.0,
        concentration_threshold=0.75,
        alpha=0.10,
        beta=0.20,
        gamma=0.10,
    )
    assert result["method"].tolist() == METHOD_ORDER
    assert result.loc[result["method"] == "Interval-Hit Greedy", "route"].item() == "direct"
    assert result.loc[result["method"] == "Interval-Hit Greedy", "effective_replacements"].item() == 0
    assert result.loc[result["method"] == "Always-DA-TPP", "route"].item() == "correction"
    assert result["shared_score_state_sha256"].nunique() == 1


def test_script_supports_direct_cli_invocation() -> None:
    archive = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable, str(archive / "analysis/run_gate_ablation_smoke.py"), "--help"],
        cwd=archive,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--development-seed" in completed.stdout


def test_cached_runtime_is_restored_only_when_current_measurement_is_absent() -> None:
    prior = pd.DataFrame({"runtime": [1.25, 1.25, 1.25]})
    assert _restore_cached_runtime(0.0, prior, "runtime") == 1.25
    assert _restore_cached_runtime(2.5, prior, "runtime") == 2.5


def test_cached_boolean_accepts_one_consistent_csv_value() -> None:
    assert _restore_cached_boolean(pd.DataFrame({"flag": [True, True]}), "flag") is True
    assert _restore_cached_boolean(pd.DataFrame({"flag": ["False", "False"]}), "flag") is False


def test_smoke_parameters_match_retained_formal_limo_run_config() -> None:
    archive = Path(__file__).resolve().parents[2]
    config_path = (
        archive
        / "baseline_snapshot/archive/experiments/reproducibility/results"
        / "paired_two_dataset_confirmation_20260712/runs/limo"
        / "energy_gated_da_tpp/seed_5/run_config.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert {name: config[name] for name in LIMO_FORMAL_PARAMS} == LIMO_FORMAL_PARAMS
