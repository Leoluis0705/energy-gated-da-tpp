from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.reproducibility.protocol_artifacts import (
    PROTOCOL_VERSION,
    score_artifact_path,
    write_dataframe_exclusive,
)
from mc_dropout_protocol import prepare_selector_uncertainty
from uncertainty_units import PredictiveMoments, interval_hit_probability_ev


def test_sigma_and_mean_are_denormalized_exactly_once() -> None:
    normalized = PredictiveMoments(
        mean=np.array([-1.0, 0.5]),
        sigma=np.array([0.2, 0.4]),
        space="normalized",
    )

    physical = normalized.to_ev(location=-2.5, scale=0.8)

    np.testing.assert_allclose(physical.mean, [-3.3, -2.1])
    np.testing.assert_allclose(physical.sigma, [0.16, 0.32])
    assert physical.space == "ev_per_atom"
    second_conversion = physical.to_ev(location=-2.5, scale=0.8)
    assert second_conversion is physical
    np.testing.assert_array_equal(second_conversion.mean, physical.mean)
    np.testing.assert_array_equal(second_conversion.sigma, physical.sigma)


def test_interval_probability_is_equivalent_in_normalized_and_ev_spaces() -> None:
    location = -2.5
    scale = 0.8
    low_ev, high_ev = -2.7, -2.1
    normalized = PredictiveMoments(
        mean=np.array([-0.3, 0.2, 0.8]),
        sigma=np.array([0.2, 0.4, 0.1]),
        space="normalized",
    )
    physical = normalized.to_ev(location=location, scale=scale)

    probability_ev = interval_hit_probability_ev(
        physical.mean,
        physical.sigma,
        low_ev,
        high_ev,
    )
    probability_normalized = interval_hit_probability_ev(
        normalized.mean,
        normalized.sigma,
        (low_ev - location) / scale,
        (high_ev - location) / scale,
    )

    np.testing.assert_allclose(probability_ev, probability_normalized, rtol=0, atol=1e-14)


def test_already_physical_moments_are_not_denormalized_again() -> None:
    physical = PredictiveMoments(
        mean=np.array([-2.2]),
        sigma=np.array([0.12]),
        space="ev_per_atom",
    )

    result = physical.to_ev(location=100.0, scale=50.0)

    assert result is physical
    np.testing.assert_array_equal(result.mean, [-2.2])
    np.testing.assert_array_equal(result.sigma, [0.12])


def test_zero_and_epsilon_variance_use_inclusive_point_mass_limit() -> None:
    means = np.array([-2.7, -2.4, -2.1, -1.9])
    sigmas = np.array([0.0, 1e-15, 0.0, 0.0])

    result = interval_hit_probability_ev(
        means,
        sigmas,
        low_ev=-2.7,
        high_ev=-2.1,
        sigma_floor_ev=0.0,
        epsilon=1e-12,
    )

    np.testing.assert_array_equal(result, [1.0, 1.0, 1.0, 0.0])


def test_positive_ev_sigma_floor_is_applied_after_denormalization() -> None:
    result = interval_hit_probability_ev(
        mean_ev=np.array([-2.4]),
        sigma_ev=np.array([0.0]),
        low_ev=-2.7,
        high_ev=-2.1,
        sigma_floor_ev=0.05,
    )

    assert 0.999 < result[0] <= 1.0


def test_invalid_normalizer_scale_and_negative_sigma_are_rejected() -> None:
    normalized = PredictiveMoments(
        mean=np.array([0.0]),
        sigma=np.array([0.1]),
        space="normalized",
    )
    with pytest.raises(ValueError, match="normalizer scale"):
        normalized.to_ev(location=-2.5, scale=0.0)
    with pytest.raises(ValueError, match="sigma"):
        interval_hit_probability_ev(
            mean_ev=np.array([-2.4]),
            sigma_ev=np.array([-0.1]),
            low_ev=-2.7,
            high_ev=-2.1,
        )


def test_shape_mismatch_and_reversed_interval_are_rejected() -> None:
    with pytest.raises(ValueError, match="same shape"):
        PredictiveMoments(
            mean=np.array([0.0, 1.0]),
            sigma=np.array([0.1]),
            space="normalized",
        )
    with pytest.raises(ValueError, match="target interval"):
        interval_hit_probability_ev(
            mean_ev=np.array([-2.4]),
            sigma_ev=np.array([0.1]),
            low_ev=-2.1,
            high_ev=-2.7,
        )


def test_selector_uncertainty_converts_sigma_to_ev_but_keeps_ev_mean() -> None:
    state = prepare_selector_uncertainty(
        deterministic_mean_ev=np.array([-2.4]),
        mc_sigma_normalized=np.array([0.5]),
        normalizer_location=-2.5,
        normalizer_scale=0.2,
        target_low_ev=-2.45,
        target_high_ev=-2.35,
        sigma_floor_ev=0.0,
    )

    np.testing.assert_allclose(state.mean_ev, [-2.4])
    np.testing.assert_allclose(state.sigma_normalized, [0.5])
    np.testing.assert_allclose(state.sigma_ev, [0.1])
    expected = interval_hit_probability_ev(
        np.array([-2.4]),
        np.array([0.1]),
        -2.45,
        -2.35,
    )
    wrong_unit_probability = interval_hit_probability_ev(
        np.array([-2.4]),
        np.array([0.5]),
        -2.45,
        -2.35,
    )
    np.testing.assert_allclose(state.interval_hit_probability, expected)
    assert not np.allclose(state.interval_hit_probability, wrong_unit_probability)
    assert state.mean_input_space == "ev_per_atom"
    assert state.sigma_input_space == "normalized"


def test_corrected_score_artifact_cannot_overwrite_legacy_or_existing_output(tmp_path) -> None:
    legacy = tmp_path / "energy_gated_da_tpp_scores_iter_1.csv"
    legacy.write_text("legacy-evidence\n", encoding="utf-8")
    corrected = score_artifact_path(
        tmp_path,
        method_name="energy_gated_da_tpp",
        iteration=1,
    )

    assert corrected != legacy
    assert PROTOCOL_VERSION in corrected.name
    write_dataframe_exclusive(pd.DataFrame({"sigma_eV": [0.1]}), corrected)
    with pytest.raises(FileExistsError):
        write_dataframe_exclusive(pd.DataFrame({"sigma_eV": [9.9]}), corrected)

    assert legacy.read_text(encoding="utf-8") == "legacy-evidence\n"
    assert pd.read_csv(corrected)["sigma_eV"].tolist() == [0.1]
