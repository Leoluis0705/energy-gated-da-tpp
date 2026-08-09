import numpy as np
import pandas as pd

from analysis.three_system.models import (
    bootstrap_binary_predictions,
    evaluate_binary_models_nested_loo,
    evaluate_energy_calibrators_loo,
    fit_energy_calibrator_predict,
)


def _binary_fixture() -> tuple[pd.DataFrame, np.ndarray]:
    x = pd.DataFrame(
        {
            "m_element": ["Cr", "Cr", "Cr", "Mn", "Mn", "Mn", "Mg", "Mg"] * 2,
            "energy_gap": [-1.0, -0.8, 0.7, -0.6, 0.5, 0.8, -0.9, 0.9] * 2,
            "volume_change": [0.01, 0.02, 0.3, 0.03, 0.25, 0.4, 0.02, 0.5] * 2,
        }
    )
    y = np.array([1, 1, 0, 1, 0, 0, 1, 0] * 2, dtype=int)
    return x, y


def test_nested_loo_returns_out_of_fold_probabilities_for_every_model_and_row():
    """Catches fitting on the held-out row or silently dropping a model."""
    x, y = _binary_fixture()
    predictions, summary = evaluate_binary_models_nested_loo(
        x,
        y,
        random_seed=19,
        model_names=(
            "regularized_logistic",
            "laplace_bayesian_logistic",
            "shallow_gradient_boosting",
            "shallow_random_forest",
        ),
    )

    assert len(predictions) == len(x) * 4
    assert predictions.groupby("model_name")["row_index"].nunique().to_dict() == {
        "laplace_bayesian_logistic": len(x),
        "regularized_logistic": len(x),
        "shallow_gradient_boosting": len(x),
        "shallow_random_forest": len(x),
    }
    assert predictions["probability"].between(0, 1).all()
    assert set(summary["model_name"]) == set(predictions["model_name"])
    assert summary["loo_log_loss"].notna().all()


def test_energy_calibration_loo_keeps_fixed_interval_and_reports_all_sources():
    """Catches moving the interval or omitting one low-fidelity energy source."""
    table = pd.DataFrame(
        {
            "candidate_id": [f"c{i}" for i in range(10)],
            "m_element": ["Cr"] * 5 + ["Mn"] * 3 + ["Mg"] * 2,
            "dft_energy": np.array(
                [-2.29, -2.28, -2.27, -2.25, -2.24, -1.87, -1.85, -1.84, -1.54, -1.51]
            ),
            "alignn": np.array(
                [-2.18, -2.17, -2.16, -2.15, -2.14, -2.03, -2.02, -2.01, -2.18, -2.17]
            ),
            "chgnet": np.array(
                [-8.07, -8.06, -8.05, -8.04, -8.03, -7.72, -7.71, -7.70, -5.46, -5.44]
            ),
            "mace": np.array(
                [-7.10, -7.09, -7.08, -7.07, -7.06, -6.86, -6.85, -6.84, -5.08, -5.06]
            ),
        }
    )

    predictions, summary = evaluate_energy_calibrators_loo(
        table,
        interval=(-2.3, -1.5),
        bootstrap_draws=200,
        random_seed=23,
    )

    assert set(summary["source"]) == {"ALIGNN", "CHGNet", "MACE-MP", "dual_MLIP_ensemble"}
    assert set(summary["interval_lower"]) == {-2.3}
    assert set(summary["interval_upper"]) == {-1.5}
    assert predictions.groupby("model_id")["candidate_id"].nunique().min() == 10
    assert summary["loo_mae_eV_atom"].notna().all()
    assert summary["loo_rmse_eV_atom"].notna().all()


def test_bootstrap_classifier_and_energy_calibrator_predict_unseen_rows():
    """Catches fitting CV-only models that cannot score the prospective pool."""
    x, y = _binary_fixture()
    binary = bootstrap_binary_predictions(
        x,
        y,
        x.iloc[:3],
        model_name="regularized_logistic",
        random_seed=31,
        draws=20,
    )
    assert binary.draws.shape == (20, 3)
    assert np.all((binary.mean >= 0) & (binary.mean <= 1))
    assert np.all(binary.standard_deviation >= 0)

    calibration = pd.DataFrame(
        {
            "candidate_id": [f"c{i}" for i in range(10)],
            "m_element": ["Cr"] * 5 + ["Mn"] * 3 + ["Mg"] * 2,
            "dft_energy": [-2.29, -2.28, -2.27, -2.25, -2.24, -1.87, -1.85, -1.84, -1.54, -1.51],
            "alignn": [-2.18, -2.17, -2.16, -2.15, -2.14, -2.03, -2.02, -2.01, -2.18, -2.17],
            "chgnet": [-8.07, -8.06, -8.05, -8.04, -8.03, -7.72, -7.71, -7.70, -5.46, -5.44],
            "mace": [-7.10, -7.09, -7.08, -7.07, -7.06, -6.86, -6.85, -6.84, -5.08, -5.06],
        }
    )
    prospective = calibration.drop(columns=["dft_energy"]).iloc[:3].copy()
    predicted = fit_energy_calibrator_predict(
        calibration,
        prospective,
        model_id="ALIGNN::composition_offset",
        interval=(-2.3, -1.5),
    )
    assert len(predicted) == 3
    assert predicted["predicted_dft_energy_mean"].notna().all()
    assert predicted["predicted_dft_energy_std"].gt(0).all()
    assert predicted["p_interval_hit"].between(0, 1).all()
