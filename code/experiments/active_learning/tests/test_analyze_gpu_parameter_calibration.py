import json
from pathlib import Path

import pandas as pd

from analysis.analyze_gpu_parameter_calibration import (
    aggregate_calibration_configs,
    build_combined_calibration_manifest,
    build_threshold_promotion_bundle,
    build_weight_promotion_bundle,
    rank_calibration_configs,
    rank_aggregated_configs,
    summarize_completed_calibration,
)
from analysis.prepare_gpu_weight_calibration import build_weight_screen_bundle
from analysis.finalize_gpu_calibration import finalize_calibration_stage


def _write_completed_run(path: Path, *, autc: float, corrections: int) -> None:
    path.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "AUTC": autc,
                "correction_rounds": corrections,
                "candidate_sequence_sha256": "a" * 64,
            }
        ]
    ).to_csv(path / "run_metrics.csv", index=False)
    (path / "status.json").write_text(
        json.dumps({"status": "DONE", "elapsed_seconds": 100.0}) + "\n",
        encoding="utf-8",
    )


def _screen_manifest(tmp_path: Path) -> Path:
    rows = []
    values = [
        ("a", 1.25, 0.60, 0.90000, 5),
        ("b", 1.00, 0.50, 0.89995, 4),
        ("c", 0.75, 0.40, 0.89980, 3),
        ("d", 0.75, 0.50, 0.80, 10),
        ("e", 0.75, 0.60, 0.79, 10),
        ("f", 1.00, 0.40, 0.78, 10),
        ("g", 1.00, 0.60, 0.77, 10),
        ("h", 1.25, 0.40, 0.76, 10),
        ("i", 1.25, 0.50, 0.75, 10),
    ]
    for config_id, m0, g0, autc, corrections in values:
        output = tmp_path / "runs" / config_id / "seed_0" / "attempt_1"
        _write_completed_run(output, autc=autc, corrections=corrections)
        remote_output = f"/remote/results/{config_id}/seed_0/attempt_1"
        command = [
            "/python",
            "/project/run.py",
            "--seed",
            "0",
            "--run-dir",
            remote_output,
            "--protocol-config",
            f"/project/configs/threshold_{config_id}.yaml",
        ]
        rows.append(
            {
                "job_id": f"gpu_cal_threshold_{config_id}_seed00",
                "dataset": "limo",
                "method": "energy_gated_da_tpp",
                "group_key": "element_system_current",
                "seed": 0,
                "K": 30,
                "config_hash": config_id * 64,
                "git_commit": "abc123",
                "gpu_id": 0,
                "status": "DONE",
                "start_time": "start",
                "end_time": "end",
                "exit_code": 0,
                "log_path": f"/remote/logs/{config_id}_seed00.log",
                "output_path": str(output),
                "sha256": "f" * 64,
                "command_json": json.dumps(command, separators=(",", ":")),
                "cwd": "/project",
                "attempt": 1,
                "pid": "",
                "failure_reason": "",
                "M0": m0,
                "G0": g0,
                "alpha": 0.1,
                "beta": 0.2,
                "gamma": 0.1,
                "calibration_stage": "threshold_seed0_screen",
            }
        )
    path = tmp_path / "screen.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_threshold_ranking_applies_preregistered_tiebreaks(tmp_path: Path) -> None:
    summary = summarize_completed_calibration(_screen_manifest(tmp_path))
    ranked = rank_calibration_configs(summary)

    assert ranked.iloc[:3]["config_id"].tolist() == ["b", "a", "c"]
    assert ranked.iloc[0]["selection_rank"] == 1
    assert ranked.loc[ranked["config_id"] == "b", "center_distance_grid_units"].item() == 0.0


def test_promotion_reuses_top_three_configs_for_seeds_one_to_four(tmp_path: Path) -> None:
    manifest = _screen_manifest(tmp_path)
    summary = summarize_completed_calibration(manifest)
    ranked = rank_calibration_configs(summary)
    output = tmp_path / "promotion.csv"

    promoted = build_threshold_promotion_bundle(
        screen_manifest_path=manifest,
        ranked_summary=ranked,
        manifest_path=output,
        remote_output_root="/remote/results",
    )

    assert len(promoted) == 12
    assert set(promoted["seed"]) == {1, 2, 3, 4}
    assert set(promoted["config_id"]) == {"a", "b", "c"}
    assert set(promoted["status"]) == {"PENDING"}
    assert promoted["job_id"].is_unique
    assert promoted["output_path"].is_unique
    for row in promoted.to_dict(orient="records"):
        command = json.loads(row["command_json"])
        assert command[command.index("--seed") + 1] == str(row["seed"])
        assert command[command.index("--run-dir") + 1] == row["output_path"]
        assert f"seed_{row['seed']}" in row["output_path"]


def test_full_development_ranking_uses_mean_autc_then_mean_corrections() -> None:
    rows = []
    for config_id, autc, corrections, distance in (
        ("near", 0.90000, 6, 0.0),
        ("fewer_corrections", 0.89995, 4, 1.0),
        ("lower_autc", 0.89970, 2, 1.0),
    ):
        for seed in range(5):
            rows.append(
                {
                    "config_id": config_id,
                    "seed": seed,
                    "M0": 1.0,
                    "G0": 0.5,
                    "alpha": 0.1,
                    "beta": 0.2,
                    "gamma": 0.1,
                    "mc_passes": 30,
                    "AUTC": autc,
                    "correction_rounds": corrections,
                    "center_distance_grid_units": distance,
                    "runtime_seconds": 100.0,
                    "candidate_sequence_sha256": f"{seed}" * 64,
                    "config_hash": config_id,
                    "git_commit": "abc",
                }
            )

    aggregated = aggregate_calibration_configs(pd.DataFrame(rows), expected_seeds=range(5))
    ranked = rank_aggregated_configs(aggregated)

    assert ranked["config_id"].tolist() == [
        "fewer_corrections",
        "near",
        "lower_autc",
    ]
    assert set(aggregated["seed_count"]) == {5}
    assert set(aggregated["seeds"]) == {"0;1;2;3;4"}


def test_combined_manifest_has_top_three_and_each_development_seed(tmp_path: Path) -> None:
    screen = _screen_manifest(tmp_path)
    ranked = rank_calibration_configs(summarize_completed_calibration(screen))
    promotion_path = tmp_path / "promotion.csv"
    promotion = build_threshold_promotion_bundle(
        screen_manifest_path=screen,
        ranked_summary=ranked,
        manifest_path=promotion_path,
        remote_output_root="/remote/results",
    )
    promotion["status"] = "DONE"
    promotion["exit_code"] = 0
    promotion.to_csv(promotion_path, index=False)
    combined_path = tmp_path / "combined.csv"

    combined = build_combined_calibration_manifest(
        screen_manifest_path=screen,
        promotion_manifest_path=promotion_path,
        ranked_seed0=ranked,
        manifest_path=combined_path,
    )

    assert len(combined) == 15
    assert combined["job_id"].is_unique
    assert combined["output_path"].is_unique
    assert set(combined["config_id"]) == {"a", "b", "c"}
    for _, block in combined.groupby("config_id"):
        assert sorted(block["seed"].astype(int).tolist()) == [0, 1, 2, 3, 4]


def test_weight_summary_uses_weight_distance_from_original_center(tmp_path: Path) -> None:
    run = tmp_path / "weight_run"
    _write_completed_run(run, autc=0.8, corrections=5)
    manifest = tmp_path / "weight.csv"
    pd.DataFrame(
        [
            {
                "job_id": "gpu_cal_weight_alpha_0p05_seed00",
                "seed": 0,
                "K": 30,
                "config_hash": "a" * 64,
                "git_commit": "abc",
                "status": "DONE",
                "exit_code": 0,
                "output_path": str(run),
                "sha256": "b" * 64,
                "M0": 1.0,
                "G0": 0.5,
                "alpha": 0.05,
                "beta": 0.2,
                "gamma": 0.1,
                "calibration_stage": "weight_seed0_screen",
            }
        ]
    ).to_csv(manifest, index=False)

    summary = summarize_completed_calibration(manifest)

    assert summary.loc[0, "config_id"] == "alpha_0p05"
    assert summary.loc[0, "center_distance_grid_units"] == 0.05


def test_weight_promotion_reuses_top_three_for_seeds_one_to_four(tmp_path: Path) -> None:
    screen = tmp_path / "weight_screen.csv"
    frame = build_weight_screen_bundle(
        project_root="/remote/project",
        output_root="/remote/results/weight",
        local_configs_root=tmp_path / "configs",
        remote_configs_root="/remote/project/configs/weight",
        manifest_path=screen,
        git_commit="abc",
        mc_passes=30,
        m0=1.0,
        g0=0.5,
    )
    ranked = pd.DataFrame(
        {
            "selection_rank": range(1, 8),
            "config_id": frame["variant_id"].tolist(),
        }
    )
    promotion_path = tmp_path / "weight_promotion.csv"

    promoted = build_weight_promotion_bundle(
        screen_manifest_path=screen,
        ranked_summary=ranked,
        manifest_path=promotion_path,
        remote_output_root="/remote/results/weight",
    )

    assert len(promoted) == 12
    assert set(promoted["seed"]) == {1, 2, 3, 4}
    assert set(promoted["config_id"]) == set(frame.iloc[:3]["variant_id"])
    assert promoted["job_id"].str.startswith("gpu_cal_weight_").all()


def test_finalize_calibration_stage_writes_ranked_audit_bundle(tmp_path: Path) -> None:
    screen = _screen_manifest(tmp_path)
    seed0_ranking = rank_calibration_configs(summarize_completed_calibration(screen))
    seed0_ranking_path = tmp_path / "seed0_ranking.csv"
    seed0_ranking.to_csv(seed0_ranking_path, index=False)
    promotion_path = tmp_path / "promotion.csv"
    promotion = build_threshold_promotion_bundle(
        screen_manifest_path=screen,
        ranked_summary=seed0_ranking,
        manifest_path=promotion_path,
        remote_output_root="/remote/results",
    )
    for index, row in promotion.iterrows():
        run = tmp_path / "promotion_runs" / str(row["config_id"]) / f"seed_{row['seed']}"
        _write_completed_run(run, autc=0.85 + index / 10000, corrections=5)
        promotion.loc[index, "output_path"] = str(run)
        promotion.loc[index, "status"] = "DONE"
        promotion.loc[index, "exit_code"] = "0"
    promotion.to_csv(promotion_path, index=False)

    result = finalize_calibration_stage(
        screen_manifest=screen,
        promotion_manifest=promotion_path,
        seed0_ranking=seed0_ranking_path,
        output_dir=tmp_path / "finalized",
    )

    assert result["selected_config_id"] in {"a", "b", "c"}
    assert (tmp_path / "finalized" / "combined_manifest.csv").is_file()
    assert (tmp_path / "finalized" / "per_seed_results.csv").is_file()
    assert (tmp_path / "finalized" / "full_ranking.csv").is_file()
    assert (tmp_path / "finalized" / "selection.json").is_file()
    sums = (tmp_path / "finalized" / "SHA256SUMS").read_text(encoding="utf-8")
    assert "combined_manifest.csv" in sums
    assert "selection.json" in sums
