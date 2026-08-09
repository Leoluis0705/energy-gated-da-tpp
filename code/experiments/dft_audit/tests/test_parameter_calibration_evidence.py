import json
from pathlib import Path

import pandas as pd

from analysis.build_parameter_calibration_evidence import build_calibration_table


ROOT = Path(__file__).resolve().parents[1]


def test_calibration_table_preserves_cohort_boundary_and_frozen_choice():
    table = build_calibration_table(
        ROOT / "results/parameter_selection/search_history.csv",
        ROOT / "configs/frozen_final_protocol.yaml",
        freeze_commit="8a7599807ac786c9bb664bb199dc5bf422db6f93",
        freeze_time="2026-07-18T03:19:53+08:00",
    )

    assert len(table) == 22
    assert set(table["selection_data_seeds"]) <= {"0", "0;1;2;3;4"}
    assert not table["selection_data_seeds"].str.contains(
        r"(?:^|;)(?:1[5-9]|2[0-4])(?:;|$)", regex=True
    ).any()

    chosen = table.loc[table["selected_final_protocol"]]
    assert len(chosen) == 1
    chosen = chosen.iloc[0]
    assert chosen["stage"] == "weight_seeds0_4"
    assert chosen["config_id"] == "gamma_0p05"
    assert chosen[["M0", "G0", "alpha", "beta", "gamma", "mc_passes"]].tolist() == [
        0.75,
        0.5,
        0.1,
        0.2,
        0.05,
        30,
    ]


def test_calibration_table_adds_verifiable_freeze_and_source_provenance():
    protocol_path = ROOT / "configs/frozen_final_protocol.yaml"
    table = build_calibration_table(
        ROOT / "results/parameter_selection/search_history.csv",
        protocol_path,
        freeze_commit="8a7599807ac786c9bb664bb199dc5bf422db6f93",
        freeze_time="2026-07-18T03:19:53+08:00",
    )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))

    assert table["frozen_protocol_sha256"].nunique() == 1
    assert table["frozen_protocol_sha256"].iat[0] == (
        "2a8d0fb5114c0e2f2457d9887ff5cfc6b8c9ff701669f473e018984855fcac84"
    )
    assert set(table["freeze_git_commit"]) == {
        "8a7599807ac786c9bb664bb199dc5bf422db6f93"
    }
    assert set(table["freeze_time_evidence"]) == {"git_commit_time"}
    assert protocol["allowed_seeds"] == list(range(15, 25))
    assert not table["used_formal_evaluation_seeds"].any()


def test_written_table_round_trips_without_manual_metric_changes(tmp_path):
    source = ROOT / "results/parameter_selection/search_history.csv"
    table = build_calibration_table(
        source,
        ROOT / "configs/frozen_final_protocol.yaml",
        freeze_commit="8a7599807ac786c9bb664bb199dc5bf422db6f93",
        freeze_time="2026-07-18T03:19:53+08:00",
    )
    output = tmp_path / "table.csv"
    table.to_csv(output, index=False)
    written = pd.read_csv(output)
    original = pd.read_csv(source)

    for column in ["stage", "config_id", "AUTC", "mean_AUTC", "correction_rounds"]:
        pd.testing.assert_series_equal(
            written[column], original[column], check_names=False, check_dtype=False
        )
