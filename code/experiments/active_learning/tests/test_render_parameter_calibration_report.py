import pandas as pd

from analysis.render_parameter_calibration_report import (
    build_search_history,
    render_parameter_calibration_report,
)


def _ranking(config_id: str, *, aggregate: bool, alpha: float = 0.1) -> pd.DataFrame:
    row = {
        "selection_rank": 1,
        "config_id": config_id,
        "M0": 1.0,
        "G0": 0.5,
        "alpha": alpha,
        "beta": 0.2,
        "gamma": 0.1,
        "mc_passes": 30,
        "center_distance_grid_units": 0.0,
    }
    if aggregate:
        row.update(
            {
                "mean_AUTC": 0.81,
                "sample_sd_AUTC": 0.01,
                "mean_correction_rounds": 20.0,
                "sample_sd_correction_rounds": 1.0,
                "seed_count": 5,
                "seeds": "0;1;2;3;4",
            }
        )
    else:
        row.update({"AUTC": 0.82, "correction_rounds": 19, "seed": 0})
    return pd.DataFrame([row])


def test_report_and_search_history_are_source_backed() -> None:
    threshold_seed0 = _ranking("threshold_center", aggregate=False)
    threshold_full = _ranking("threshold_center", aggregate=True)
    weight_seed0 = _ranking("alpha_0p05", aggregate=False, alpha=0.05)
    weight_full = _ranking("alpha_0p05", aggregate=True, alpha=0.05)

    history = build_search_history(
        threshold_seed0=threshold_seed0,
        threshold_full=threshold_full,
        weight_seed0=weight_seed0,
        weight_full=weight_full,
    )
    report = render_parameter_calibration_report(
        threshold_seed0=threshold_seed0,
        threshold_full=threshold_full,
        weight_seed0=weight_seed0,
        weight_full=weight_full,
        selected_mc_passes=30,
        source_sha256={"weight_full": "a" * 64},
    )

    assert set(history["stage"]) == {
        "threshold_seed0",
        "threshold_seeds0_4",
        "weight_seed0",
        "weight_seeds0_4",
    }
    assert "Final frozen choice" in report
    assert "`alpha = 0.05`" in report
    assert "development seeds 0-4" in report
    assert "aaaaaaaaaaaaaaaa" in report

