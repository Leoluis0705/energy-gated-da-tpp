import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from active_learning_energy_gate_ablation import select_baseline_indices
from experiments.reproducibility.formal_protocol import load_formal_protocol
from experiments.reproducibility.interval_task_protocol import (
    IntervalTaskError,
    load_initial_set_ids,
    load_interval_task,
)
from experiments.reproducibility import run_paired_dataset_job as runner
from analysis.build_mn_mg_gpu_manifest import build_manifest_rows


def test_gpu_runner_can_be_invoked_as_a_direct_server_script():
    project = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(project / "experiments/reproducibility/run_paired_dataset_job.py"),
            "--help",
        ],
        cwd=project,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def write_task(tmp_path, **overrides):
    payload = {
        "task_version": "mn_mg_interval_gpu_v1",
        "task_id": "mn_proxy_w0p2",
        "base_dataset": "limo",
        "target_low": -2.1,
        "target_high": -1.9,
        "target_count": 126,
        "pool_size": 640,
        "budget": 320,
        "batch_size": 4,
        "rounds": 79,
        "initial_set_size": 4,
        "initial_sets_relative_path": "configs/mn_mg_initial_sets.csv",
        "checkpoints": [80, 160, 240, 320],
        "label_source": "frozen ALIGNN proxy formation energy",
        "hidden_evaluability_role": "post_selection_only",
        "frozen": True,
    }
    payload.update(overrides)
    path = tmp_path / "task.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def write_protocol(tmp_path):
    payload = {
        "protocol_version": "egdatpp_psfix_v1",
        "phase": "interval_robustness",
        "dataset": "limo",
        "allowed_seeds": list(range(101, 111)),
        "allowed_methods": [
            "energy_gated_da_tpp",
            "predicted_target_greedy",
            "explore",
            "mc_dropout",
            "gradient_norm_hybrid",
            "random_sampling",
            "always_da_tpp",
            "group_only_gate",
            "margin_only_gate",
        ],
        "mc_passes": 30,
        "M0": 0.75,
        "G0": 0.50,
        "alpha": 0.10,
        "beta": 0.20,
        "gamma": 0.05,
        "group_key_mode": "element_system_current",
        "group_key_map_relative_path": None,
        "frozen": True,
    }
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_interval_task_overrides_only_task_dimensions(tmp_path):
    task = load_interval_task(write_task(tmp_path))
    resolved = task.apply(runner.dataset_configs()["limo"])

    assert resolved.target_interval == (-2.1, -1.9)
    assert resolved.target_count == 126
    assert resolved.batch_size == 4
    assert resolved.budget == 320
    assert resolved.rounds == 79
    assert resolved.initial_labeled_set == "frozen_seed_specific_4"
    assert task.sha256


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"target_high": -2.2}, "target_low"),
        ({"rounds": 80}, "rounds"),
        ({"initial_set_size": 0}, "initial_set_size"),
        ({"frozen": False}, "must be frozen"),
    ],
)
def test_interval_task_rejects_invalid_freeze(tmp_path, overrides, message):
    with pytest.raises(IntervalTaskError, match=message):
        load_interval_task(write_task(tmp_path, **overrides))


def test_interval_protocol_accepts_all_nine_methods_and_seed_cohort(tmp_path):
    protocol = load_formal_protocol(write_protocol(tmp_path))
    for method in protocol.allowed_methods:
        protocol.resolve_dataset_config(
            runner.dataset_configs()["limo"], method=method, seed=101
        )
    assert runner.parse_experiment_seed("110") == 110
    assert runner.parse_experiment_seed("115") == 115
    with pytest.raises(Exception, match="declared"):
        runner.parse_experiment_seed("116")


def test_initial_set_loader_checks_seed_hash_and_pool_membership(tmp_path):
    ids = ["a", "b", "c", "d"]
    digest = runner.candidate_order_digest(ids)
    path = tmp_path / "initial.csv"
    pd.DataFrame(
        {
            "seed": [101] * 4,
            "candidate_id": ids,
            "energy_stratum": [0, 1, 2, 3],
            "initial_set_sha256": [digest] * 4,
        }
    ).to_csv(path, index=False)

    assert load_initial_set_ids(path, seed=101, pool_ids=ids + ["e"], expected_size=4) == ids

    bad = pd.read_csv(path)
    bad.loc[3, "candidate_id"] = "missing"
    bad.to_csv(path, index=False)
    with pytest.raises(IntervalTaskError, match="pool"):
        load_initial_set_ids(path, seed=101, pool_ids=ids + ["e"], expected_size=4)


def test_materialize_initial_set_removes_candidates_and_records_round_zero(tmp_path):
    cif_dir = tmp_path / "cifs"
    train_dir = tmp_path / "train"
    cif_dir.mkdir()
    train_dir.mkdir()
    ids = ["a", "b", "c", "d", "e"]
    pd.DataFrame({"id": ids, "value": np.arange(5.0)}).to_csv(
        cif_dir / "id_prop.csv", header=False, index=False
    )
    for candidate_id in ids:
        (cif_dir / f"{candidate_id}.cif").write_text(candidate_id, encoding="utf-8")
    oracle = pd.DataFrame(
        {
            "id": ids,
            "oracle_value": [-2.0, -1.0, -2.1, -1.5, -1.4],
            "target_label": [1, 0, 1, 0, 0],
        }
    )

    runner.materialize_initial_set(
        initial_ids=ids[:4],
        oracle=oracle,
        cif_dir=cif_dir,
        train_dir=train_dir,
        history_path=tmp_path / "history.csv",
        method="energy_gated_da_tpp",
        seed=101,
    )

    remaining = pd.read_csv(cif_dir / "id_prop.csv", header=None)[0].tolist()
    trained = pd.read_csv(train_dir / "id_prop.csv", header=None)[0].tolist()
    history = pd.read_csv(tmp_path / "history.csv")
    assert remaining == ["e"]
    assert trained == ids[:4]
    assert history.columns.tolist() == runner.SELECTION_HISTORY_COLUMNS
    assert history["iteration"].tolist() == [0, 0, 0, 0]
    appended = {column: None for column in runner.SELECTION_HISTORY_COLUMNS}
    appended.update({"id": "e", "iteration": 1, "selection_method": "test"})
    pd.DataFrame([appended]).to_csv(tmp_path / "history.csv", mode="a", header=False, index=False)
    combined = pd.read_csv(tmp_path / "history.csv")
    assert combined.shape == (5, len(runner.SELECTION_HISTORY_COLUMNS))
    runner.update_selected_history(tmp_path / "history.csv", oracle, "energy_gated_da_tpp")
    assert pd.read_csv(tmp_path / "history.csv")["target_label"].tolist() == [1, 0, 1, 0, 0]


def test_build_summary_counts_initial_set_in_query_budget(tmp_path):
    oracle = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d", "e", "f", "g", "h"],
            "oracle_value": [-2.0] * 8,
            "target_label": [1, 0, 1, 0, 1, 0, 0, 1],
        }
    )
    pd.DataFrame(
        {
            "id": list("abcdefgh"),
            "iteration": [0] * 4 + [1] * 4,
            "target_label": oracle["target_label"],
        }
    ).to_csv(tmp_path / "al_history.csv", index=False)
    pd.DataFrame(
        {
            "iteration": [1],
            "margin_score": [0.5],
            "group_concentration": [0.75],
            "mode": ["diversity_aware"],
        }
    ).to_csv(runner.trace_artifact_path(tmp_path), index=False)
    pd.DataFrame(
        {
            "round": [1],
            "selected_unique_groups": [3],
            "selected_group_repetition_rate": [0.25],
            "correction_replacement_count": [1],
            "correction_target_gain": [1],
        }
    ).to_csv(tmp_path / "round_diagnostics.csv", index=False)

    trajectory, metrics = runner.build_summary(
        tmp_path,
        "limo",
        "energy_gated_da_tpp",
        101,
        oracle,
        budget=8,
        batch_size=4,
        initial_set_size=4,
        checkpoints=(4, 8),
    )

    assert trajectory["oracle_evaluations"].tolist() == [4, 8]
    assert trajectory["cumulative_target_count"].tolist() == [2, 4]
    assert metrics.iloc[0]["recovery_at_4"] == 2
    assert metrics.iloc[0]["recovery_at_8"] == 4


def selector_inputs():
    ids = ["a", "b", "c", "d", "e", "f"]
    p_hit = np.array([0.90, 0.80, 0.70, 0.60, 0.50, 0.40])
    uncertainty = np.array([0.10, 0.95, 0.20, 0.85, 0.30, 0.40])
    features = np.array(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.0, 1.0],
            [0.01, 0.99],
            [-1.0, 0.0],
            [0.0, -1.0],
        ]
    )
    similarity = features / np.linalg.norm(features, axis=1, keepdims=True)
    gradient_proxy = np.array([0.1, 0.2, 0.95, 0.85, 0.3, 0.4])
    return ids, p_hit, uncertainty, similarity, gradient_proxy


def test_mc_dropout_baseline_selects_highest_corrected_uncertainty():
    ids, p_hit, uncertainty, similarity, gradient_proxy = selector_inputs()
    selected, _, mode = select_baseline_indices(
        "mc_dropout", ids, p_hit, uncertainty, similarity, gradient_proxy,
        batch_size=2, experiment_seed=101, iteration=1,
    )
    assert selected == [1, 3]
    assert mode == "mc_dropout"


def test_random_baseline_is_reproducible_but_seed_specific():
    args = selector_inputs()
    first = select_baseline_indices(
        "random_sampling", *args, batch_size=3, experiment_seed=101, iteration=1
    )[0]
    repeat = select_baseline_indices(
        "random_sampling", *args, batch_size=3, experiment_seed=101, iteration=1
    )[0]
    other = select_baseline_indices(
        "random_sampling", *args, batch_size=3, experiment_seed=102, iteration=1
    )[0]
    assert first == repeat
    assert first != other
    assert len(first) == len(set(first)) == 3


def test_gradient_hybrid_uses_half_gradient_proxy_then_greedy_without_duplicates():
    ids, p_hit, uncertainty, similarity, gradient_proxy = selector_inputs()
    selected, _, mode = select_baseline_indices(
        "gradient_norm_hybrid", ids, p_hit, uncertainty, similarity, gradient_proxy,
        batch_size=4, experiment_seed=101, iteration=1,
    )
    assert selected[:2] == [2, 3]
    assert selected[2:] == [0, 1]
    assert len(selected) == len(set(selected)) == 4
    assert mode == "gradient_norm_hybrid"


def test_explore_combines_kmeanspp_frontier_and_greedy_remainder():
    ids, p_hit, uncertainty, similarity, gradient_proxy = selector_inputs()
    selected, _, mode = select_baseline_indices(
        "explore", ids, p_hit, uncertainty, similarity, gradient_proxy,
        batch_size=4, experiment_seed=101, iteration=1,
    )
    assert len(selected) == len(set(selected)) == 4
    assert mode == "explore_kmeanspp_plus_greedy"
    assert any(index in selected[:2] for index in (2, 3, 4, 5))


def test_full_gpu_manifest_contains_complete_unique_180_job_grid():
    rows = build_manifest_rows(
        project_root="/project",
        execution_root="/execution",
        python="/venv/bin/python",
        smoke=False,
        protocol_sha256="abc123",
    )
    assert len(rows) == 180
    assert len({row["job_id"] for row in rows}) == 180
    assert len({row["output_path"] for row in rows}) == 180
    assert {row["gpu_id"] for row in rows} == {"0", "1"}
    assert {row["status"] for row in rows} == {"PENDING"}
    assert all("--task-config" in json.loads(row["command_json"]) for row in rows)
