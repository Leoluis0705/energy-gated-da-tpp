"""Small-sample classifiers and multi-fidelity formation-energy calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression, Ridge
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BINARY_MODEL_NAMES = (
    "regularized_logistic",
    "laplace_bayesian_logistic",
    "shallow_gradient_boosting",
    "shallow_random_forest",
)


def _preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    categorical = [
        column
        for column in frame.columns
        if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    numeric = [column for column in frame.columns if column not in categorical]
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "one_hot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


class LaplaceLogisticClassifier(ClassifierMixin, BaseEstimator):
    """L2 logistic MAP with a Laplace posterior probability correction."""

    def __init__(self, C: float = 1.0, random_state: int = 0):
        self.C = C
        self.random_state = random_state

    def fit(self, x: np.ndarray, y: np.ndarray):
        self.model_ = LogisticRegression(
            C=self.C,
            solver="lbfgs",
            max_iter=2000,
            random_state=self.random_state,
        ).fit(x, y)
        design = np.column_stack([np.ones(len(x)), np.asarray(x, dtype=float)])
        beta = np.concatenate([self.model_.intercept_, self.model_.coef_.ravel()])
        probability = expit(design @ beta)
        weights = probability * (1.0 - probability)
        precision = design.T @ (design * weights[:, None])
        regularization = np.eye(design.shape[1]) / float(self.C)
        regularization[0, 0] = 1e-8
        self.posterior_covariance_ = np.linalg.pinv(precision + regularization)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        design = np.column_stack([np.ones(len(x)), np.asarray(x, dtype=float)])
        beta = np.concatenate(
            [self.model_.intercept_, self.model_.coef_.ravel()]
        )
        mean = design @ beta
        variance = np.einsum(
            "ij,jk,ik->i", design, self.posterior_covariance_, design
        ).clip(min=0)
        corrected = expit(mean / np.sqrt(1.0 + np.pi * variance / 8.0))
        return np.column_stack([1.0 - corrected, corrected])


def _classifier_pipeline(
    frame: pd.DataFrame,
    model_name: str,
    parameter: float,
    random_seed: int,
) -> Pipeline:
    if model_name == "regularized_logistic":
        classifier = LogisticRegression(
            C=float(parameter),
            class_weight="balanced",
            solver="liblinear",
            max_iter=2000,
            random_state=random_seed,
        )
    elif model_name == "laplace_bayesian_logistic":
        classifier = LaplaceLogisticClassifier(
            C=float(parameter), random_state=random_seed
        )
    elif model_name == "shallow_gradient_boosting":
        classifier = GradientBoostingClassifier(
            n_estimators=int(parameter),
            learning_rate=0.05,
            max_depth=1,
            min_samples_leaf=2,
            random_state=random_seed,
        )
    elif model_name == "shallow_random_forest":
        classifier = RandomForestClassifier(
            n_estimators=50,
            max_depth=int(parameter),
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=random_seed,
            n_jobs=1,
        )
    else:
        raise ValueError(f"unknown binary model: {model_name}")
    return Pipeline([("preprocess", _preprocessor(frame)), ("model", classifier)])


def _parameter_grid(model_name: str) -> tuple[float, ...]:
    return {
        "regularized_logistic": (0.05, 0.2, 1.0, 5.0),
        "laplace_bayesian_logistic": (0.05, 0.2, 1.0),
        "shallow_gradient_boosting": (10, 25, 50),
        "shallow_random_forest": (1, 2, 3),
    }[model_name]


def _inner_loo_parameter(
    x: pd.DataFrame,
    y: np.ndarray,
    model_name: str,
    random_seed: int,
) -> float:
    if len(np.unique(y)) < 2:
        return _parameter_grid(model_name)[0]
    splitter = LeaveOneOut()
    scores = []
    for parameter in _parameter_grid(model_name):
        observed: list[int] = []
        probabilities: list[float] = []
        for train, test in splitter.split(x):
            y_train = y[train]
            if len(np.unique(y_train)) < 2:
                probability = float(np.mean(y_train))
            else:
                model = _classifier_pipeline(
                    x.iloc[train],
                    model_name,
                    parameter,
                    random_seed,
                )
                model.fit(x.iloc[train], y_train)
                probability = float(model.predict_proba(x.iloc[test])[:, 1][0])
            observed.append(int(y[test][0]))
            probabilities.append(float(np.clip(probability, 1e-6, 1 - 1e-6)))
        scores.append((log_loss(observed, probabilities, labels=[0, 1]), parameter))
    return min(scores, key=lambda item: (item[0], float(item[1])))[1]


def evaluate_binary_models_nested_loo(
    x: pd.DataFrame,
    y: Iterable[int],
    *,
    random_seed: int,
    model_names: tuple[str, ...] = BINARY_MODEL_NAMES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return strictly out-of-fold probabilities and model-level metrics."""
    y_array = np.asarray(list(y), dtype=int)
    if len(x) != len(y_array):
        raise ValueError("feature and label row counts differ")
    if len(np.unique(y_array)) != 2:
        raise ValueError("binary model comparison requires both classes")

    rows: list[dict[str, object]] = []
    outer = LeaveOneOut()
    for model_name in model_names:
        for fold, (train, test) in enumerate(outer.split(x)):
            parameter = _inner_loo_parameter(
                x.iloc[train],
                y_array[train],
                model_name,
                random_seed + fold,
            )
            model = _classifier_pipeline(
                x.iloc[train],
                model_name,
                parameter,
                random_seed + fold,
            )
            model.fit(x.iloc[train], y_array[train])
            probability = float(model.predict_proba(x.iloc[test])[:, 1][0])
            rows.append(
                {
                    "model_name": model_name,
                    "row_index": int(test[0]),
                    "observed_label": int(y_array[test][0]),
                    "probability": float(np.clip(probability, 1e-6, 1 - 1e-6)),
                    "selected_parameter": float(parameter),
                    "outer_fold": fold,
                }
            )
    predictions = pd.DataFrame(rows)
    summaries = []
    for model_name, group in predictions.groupby("model_name", sort=True):
        observed = group["observed_label"].to_numpy()
        probability = group["probability"].to_numpy()
        predicted = (probability >= 0.5).astype(int)
        summaries.append(
            {
                "model_name": model_name,
                "n": len(group),
                "positives": int(observed.sum()),
                "loo_roc_auc": float(roc_auc_score(observed, probability)),
                "loo_balanced_accuracy": float(
                    balanced_accuracy_score(observed, predicted)
                ),
                "loo_brier_score": float(brier_score_loss(observed, probability)),
                "loo_log_loss": float(
                    log_loss(observed, probability, labels=[0, 1])
                ),
            }
        )
    summary = pd.DataFrame(summaries).sort_values(
        ["loo_log_loss", "loo_brier_score", "model_name"],
        kind="mergesort",
    )
    return predictions, summary.reset_index(drop=True)


def fit_binary_pipeline(
    x: pd.DataFrame,
    y: Iterable[int],
    *,
    model_name: str,
    random_seed: int,
) -> Pipeline:
    y_array = np.asarray(list(y), dtype=int)
    parameter = _inner_loo_parameter(x, y_array, model_name, random_seed)
    model = _classifier_pipeline(x, model_name, parameter, random_seed)
    return model.fit(x, y_array)


@dataclass(frozen=True)
class BootstrapPrediction:
    mean: np.ndarray
    standard_deviation: np.ndarray
    draws: np.ndarray


def bootstrap_binary_predictions(
    x_train: pd.DataFrame,
    y_train: Iterable[int],
    x_predict: pd.DataFrame,
    *,
    model_name: str,
    random_seed: int,
    draws: int = 200,
    fixed_parameter: float | None = None,
) -> BootstrapPrediction:
    """Fit class-stratified bootstrap models and retain predictive uncertainty."""
    y_array = np.asarray(list(y_train), dtype=int)
    rng = np.random.default_rng(random_seed)
    class_indices = {
        value: np.flatnonzero(y_array == value) for value in np.unique(y_array)
    }
    parameter = (
        _inner_loo_parameter(x_train, y_array, model_name, random_seed)
        if fixed_parameter is None
        else fixed_parameter
    )
    predictions = []
    for index in range(draws):
        sampled = np.concatenate(
            [
                rng.choice(indices, size=len(indices), replace=True)
                for indices in class_indices.values()
            ]
        )
        rng.shuffle(sampled)
        sampled_x = x_train.iloc[sampled].reset_index(drop=True)
        model = _classifier_pipeline(
            sampled_x,
            model_name,
            parameter,
            random_seed + index,
        )
        model.fit(sampled_x, y_array[sampled])
        predictions.append(model.predict_proba(x_predict)[:, 1])
    matrix = np.vstack(predictions)
    return BootstrapPrediction(
        mean=matrix.mean(axis=0),
        standard_deviation=matrix.std(axis=0, ddof=1),
        draws=matrix,
    )


def _energy_design(
    frame: pd.DataFrame,
    source_columns: tuple[str, ...],
) -> tuple[np.ndarray, list[str]]:
    numeric = frame.loc[:, source_columns].astype(float).to_numpy()
    elements = pd.get_dummies(frame["m_element"], prefix="M", dtype=float)
    design = np.column_stack([numeric, elements.to_numpy()])
    names = [*source_columns, *elements.columns.tolist()]
    return design, names


def _composition_offset_prediction(
    train: pd.DataFrame,
    test: pd.DataFrame,
    low_column: str,
) -> tuple[float, float, np.ndarray]:
    residual = train["dft_energy"].to_numpy() - train[low_column].to_numpy()
    element = str(test["m_element"].iloc[0])
    same = train["m_element"].astype(str) == element
    local = residual[same.to_numpy()]
    if len(local) == 0:
        local = residual
    offset = float(np.mean(local))
    prediction = float(test[low_column].iloc[0] + offset)
    train_prediction = train[low_column].to_numpy() + np.array(
        [
            np.mean(
                residual[
                    (train["m_element"].astype(str) == str(value)).to_numpy()
                ]
            )
            for value in train["m_element"]
        ]
    )
    train_residual = train["dft_energy"].to_numpy() - train_prediction
    standard_deviation = float(
        np.std(train_residual, ddof=1) if len(train_residual) > 1 else 0.1
    )
    return prediction, max(standard_deviation, 1e-6), train_residual


def evaluate_energy_calibrators_loo(
    table: pd.DataFrame,
    *,
    interval: tuple[float, float],
    bootstrap_draws: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare low-complexity energy maps without moving the frozen interval."""
    required = {
        "candidate_id",
        "m_element",
        "dft_energy",
        "alignn",
        "chgnet",
        "mace",
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"energy calibration table is missing: {sorted(missing)}")
    data = table.copy().reset_index(drop=True)
    data["dual_mlip"] = (data["chgnet"] + data["mace"]) / 2.0
    source_map = {
        "ALIGNN": "alignn",
        "CHGNet": "chgnet",
        "MACE-MP": "mace",
        "dual_MLIP_ensemble": "dual_mlip",
    }
    rng = np.random.default_rng(random_seed)
    predictions: list[dict[str, object]] = []
    splitter = LeaveOneOut()
    for source, column in source_map.items():
        for method in ("composition_offset", "ridge_calibration"):
            model_id = f"{source}::{method}"
            for fold, (train_index, test_index) in enumerate(splitter.split(data)):
                train = data.iloc[train_index].copy()
                test = data.iloc[test_index].copy()
                if method == "composition_offset":
                    mean, standard_deviation, residuals = (
                        _composition_offset_prediction(train, test, column)
                    )
                else:
                    train_design, _ = _energy_design(train, (column,))
                    test_design, _ = _energy_design(
                        pd.concat([train, test], ignore_index=True),
                        (column,),
                    )
                    test_design = test_design[-1:]
                    model = Ridge(alpha=10.0).fit(
                        train_design, train["dft_energy"].to_numpy()
                    )
                    mean = float(model.predict(test_design)[0])
                    residuals = (
                        train["dft_energy"].to_numpy()
                        - model.predict(train_design)
                    )
                    standard_deviation = float(
                        max(np.std(residuals, ddof=1), 1e-6)
                    )
                sampled = mean + rng.choice(
                    residuals if len(residuals) else np.array([0.0]),
                    size=bootstrap_draws,
                    replace=True,
                )
                predictions.append(
                    {
                        "model_id": model_id,
                        "source": source,
                        "method": method,
                        "candidate_id": test["candidate_id"].iloc[0],
                        "m_element": test["m_element"].iloc[0],
                        "observed_dft_energy_eV_atom": float(
                            test["dft_energy"].iloc[0]
                        ),
                        "predicted_dft_energy_eV_atom": mean,
                        "prediction_standard_deviation_eV_atom": standard_deviation,
                        "prediction_interval_lower_95": float(
                            np.quantile(sampled, 0.025)
                        ),
                        "prediction_interval_upper_95": float(
                            np.quantile(sampled, 0.975)
                        ),
                        "outer_fold": fold,
                    }
                )

    model_id = "dual_MLIP_ensemble::bayesian_ridge_ensemble"
    for fold, (train_index, test_index) in enumerate(splitter.split(data)):
        train = data.iloc[train_index].copy()
        test = data.iloc[test_index].copy()
        combined = pd.concat([train, test], ignore_index=True)
        full_design, _ = _energy_design(
            combined, ("alignn", "chgnet", "mace")
        )
        train_design = full_design[:-1]
        test_design = full_design[-1:]
        model = BayesianRidge().fit(
            train_design, train["dft_energy"].to_numpy()
        )
        mean_array, standard_deviation_array = model.predict(
            test_design, return_std=True
        )
        mean = float(mean_array[0])
        standard_deviation = float(max(standard_deviation_array[0], 1e-6))
        sampled = rng.normal(mean, standard_deviation, size=bootstrap_draws)
        predictions.append(
            {
                "model_id": model_id,
                "source": "dual_MLIP_ensemble",
                "method": "bayesian_ridge_ensemble",
                "candidate_id": test["candidate_id"].iloc[0],
                "m_element": test["m_element"].iloc[0],
                "observed_dft_energy_eV_atom": float(test["dft_energy"].iloc[0]),
                "predicted_dft_energy_eV_atom": mean,
                "prediction_standard_deviation_eV_atom": standard_deviation,
                "prediction_interval_lower_95": float(np.quantile(sampled, 0.025)),
                "prediction_interval_upper_95": float(np.quantile(sampled, 0.975)),
                "outer_fold": fold,
            }
        )

    prediction_frame = pd.DataFrame(predictions)
    summaries: list[dict[str, object]] = []
    for model_id, group in prediction_frame.groupby("model_id", sort=True):
        observed = group["observed_dft_energy_eV_atom"].to_numpy()
        predicted = group["predicted_dft_energy_eV_atom"].to_numpy()
        error = predicted - observed
        summaries.append(
            {
                "model_id": model_id,
                "source": group["source"].iloc[0],
                "method": group["method"].iloc[0],
                "n": len(group),
                "loo_mae_eV_atom": float(mean_absolute_error(observed, predicted)),
                "loo_rmse_eV_atom": float(
                    mean_squared_error(observed, predicted) ** 0.5
                ),
                "loo_bias_eV_atom": float(np.mean(error)),
                "prediction_interval_coverage_95": float(
                    np.mean(
                        (
                            observed
                            >= group["prediction_interval_lower_95"].to_numpy()
                        )
                        & (
                            observed
                            <= group["prediction_interval_upper_95"].to_numpy()
                        )
                    )
                ),
                "interval_lower": float(interval[0]),
                "interval_upper": float(interval[1]),
            }
        )
    summary = pd.DataFrame(summaries).sort_values(
        ["loo_mae_eV_atom", "loo_rmse_eV_atom", "model_id"],
        kind="mergesort",
    )
    return prediction_frame, summary.reset_index(drop=True)


def select_binary_parameter(
    x: pd.DataFrame,
    y: Iterable[int],
    *,
    model_name: str,
    random_seed: int,
) -> float:
    """Select one frozen hyperparameter from the initial real labels only."""
    return float(
        _inner_loo_parameter(
            x,
            np.asarray(list(y), dtype=int),
            model_name,
            random_seed,
        )
    )


def fit_energy_calibrator_predict(
    calibration: pd.DataFrame,
    prospective: pd.DataFrame,
    *,
    model_id: str,
    interval: tuple[float, float],
) -> pd.DataFrame:
    """Fit one frozen calibrator on all quantitative DFT points and score candidates."""
    train = calibration.copy().reset_index(drop=True)
    predict = prospective.copy().reset_index(drop=True)
    for frame in (train, predict):
        frame["dual_mlip"] = (frame["chgnet"] + frame["mace"]) / 2.0
    source, method = model_id.split("::", maxsplit=1)
    source_column = {
        "ALIGNN": "alignn",
        "CHGNet": "chgnet",
        "MACE-MP": "mace",
        "dual_MLIP_ensemble": "dual_mlip",
    }[source]

    if method == "composition_offset":
        global_residual = (
            train["dft_energy"].to_numpy() - train[source_column].to_numpy()
        )
        means: list[float] = []
        standard_deviations: list[float] = []
        for row in predict.itertuples(index=False):
            element = str(row.m_element)
            mask = train["m_element"].astype(str) == element
            residual = (
                train.loc[mask, "dft_energy"].to_numpy()
                - train.loc[mask, source_column].to_numpy()
            )
            if len(residual) == 0:
                residual = global_residual
            means.append(
                float(getattr(row, source_column) + np.mean(residual))
            )
            standard_deviations.append(
                float(max(np.std(residual, ddof=1), 0.02))
                if len(residual) > 1
                else 0.10
            )
    elif method == "ridge_calibration":
        combined = pd.concat([train, predict], ignore_index=True)
        design, _ = _energy_design(combined, (source_column,))
        train_design = design[: len(train)]
        predict_design = design[len(train) :]
        model = Ridge(alpha=10.0).fit(
            train_design, train["dft_energy"].to_numpy()
        )
        means = model.predict(predict_design).astype(float).tolist()
        residual = train["dft_energy"].to_numpy() - model.predict(train_design)
        standard_deviation = float(max(np.std(residual, ddof=1), 0.02))
        standard_deviations = [standard_deviation] * len(predict)
    elif method == "bayesian_ridge_ensemble":
        combined = pd.concat([train, predict], ignore_index=True)
        design, _ = _energy_design(combined, ("alignn", "chgnet", "mace"))
        train_design = design[: len(train)]
        predict_design = design[len(train) :]
        model = BayesianRidge().fit(
            train_design, train["dft_energy"].to_numpy()
        )
        mean_array, standard_deviation_array = model.predict(
            predict_design, return_std=True
        )
        means = mean_array.astype(float).tolist()
        standard_deviations = np.maximum(
            standard_deviation_array.astype(float), 0.02
        ).tolist()
    else:
        raise ValueError(f"unknown energy calibration model: {model_id}")

    result = predict.loc[:, ["candidate_id", "m_element"]].copy()
    result["predicted_dft_energy_mean"] = means
    result["predicted_dft_energy_std"] = standard_deviations
    lower, upper = map(float, interval)
    mean_array = result["predicted_dft_energy_mean"].to_numpy()
    std_array = result["predicted_dft_energy_std"].to_numpy()
    result["p_interval_hit"] = norm.cdf(
        (upper - mean_array) / std_array
    ) - norm.cdf((lower - mean_array) / std_array)
    result["energy_calibration_model_id"] = model_id
    return result
