"""Paired pseudo-oracles, gate semantics, and prospective-union freezing."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from analysis.three_system.models import (
    bootstrap_binary_predictions,
    select_binary_parameter,
)

ALL_METHODS = (
    "random_sampling",
    "predicted_target_greedy",
    "joint_qualified_greedy",
    "dft_evaluable_greedy",
    "mc_uncertainty_only",
    "explore_core_set",
    "gradient_norm_hybrid",
    "mlip_energy_greedy",
    "composition_only",
    "group_gated_da_tpp",
    "always_correction",
    "group_only",
    "margin_only",
    "full_gate",
)


def make_paired_pseudo_oracle(scores: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    """Draw one candidate-level pseudo truth shared by every policy."""
    required = {
        "candidate_id",
        "p_dft_evaluable",
        "predicted_dft_energy_mean",
        "predicted_dft_energy_std",
    }
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"candidate scores are missing: {sorted(missing)}")
    rng = np.random.default_rng(seed)
    result = scores.loc[:, sorted(required)].copy()
    probability = result["p_dft_evaluable"].astype(float).clip(0, 1).to_numpy()
    result["pseudo_dft_evaluable"] = rng.binomial(1, probability)
    result["pseudo_dft_energy_eV_atom"] = rng.normal(
        result["predicted_dft_energy_mean"].astype(float).to_numpy(),
        result["predicted_dft_energy_std"].astype(float).clip(lower=1e-6).to_numpy(),
    )
    return result.sort_values("candidate_id", kind="mergesort").reset_index(drop=True)


def gate_uses_correction(
    mode: str,
    *,
    margin: float,
    concentration: float,
    margin_threshold: float = 0.10,
    concentration_threshold: float = 0.50,
) -> bool:
    """Return correction-route semantics for the frozen ablations."""
    if mode == "always_correction":
        return True
    if mode == "group_only":
        return concentration > concentration_threshold
    if mode == "margin_only":
        return margin < margin_threshold
    if mode in {"full_gate", "group_gated_da_tpp"}:
        return (margin < margin_threshold) and (
            concentration > concentration_threshold
        )
    if mode in {
        "predicted_target_greedy",
        "joint_qualified_greedy",
        "dft_evaluable_greedy",
    }:
        return False
    raise ValueError(f"unknown gate mode: {mode}")


def freeze_prospective_union(
    early_selections: pd.DataFrame,
    *,
    maximum_candidates: int,
    required_elements: tuple[str, ...],
    composition_quotas: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Deterministically cover methods and compositions within the hard cap."""
    required = {
        "candidate_id",
        "m_element",
        "structure_matcher_cluster",
        "atom_count",
        "space_group_number",
        "method",
        "seed",
        "query",
    }
    missing = required - set(early_selections.columns)
    if missing:
        raise ValueError(f"early selections are missing: {sorted(missing)}")
    if not set(required_elements) <= set(early_selections["m_element"]):
        raise ValueError("early selections do not cover all required compositions")

    grouped_rows = []
    for candidate_id, group in early_selections.groupby("candidate_id", sort=True):
        methods = sorted(set(group["method"].astype(str)))
        row = group.sort_values(
            ["query", "method", "seed"], kind="mergesort"
        ).iloc[0]
        grouped_rows.append(
            {
                "candidate_id": candidate_id,
                "m_element": row["m_element"],
                "structure_matcher_cluster": row["structure_matcher_cluster"],
                "atom_count": float(row["atom_count"]),
                "space_group_number": row["space_group_number"],
                "represented_methods": "|".join(methods),
                "baseline_support_count": len(methods),
                "median_first_query": float(group["query"].median()),
                "supporting_seed_count": int(group["seed"].nunique()),
            }
        )
    candidates = pd.DataFrame(grouped_rows)
    selected: list[str] = []

    selected_clusters: set[str] = set()

    def rank(frame: pd.DataFrame, uncovered: set[str]) -> pd.DataFrame:
        ranked = frame.copy()
        ranked["_new_coverage"] = ranked["represented_methods"].map(
            lambda text: len(set(text.split("|")) & uncovered)
        )
        ranked["_new_structure_cluster"] = ~ranked[
            "structure_matcher_cluster"
        ].astype(str).isin(selected_clusters)
        return ranked.sort_values(
            [
                "_new_coverage",
                "_new_structure_cluster",
                "baseline_support_count",
                "median_first_query",
                "atom_count",
                "space_group_number",
                "candidate_id",
            ],
            ascending=[False, False, False, True, True, True, True],
            kind="mergesort",
        )

    all_methods = set(early_selections["method"].astype(str))
    uncovered = set(all_methods)
    selected_composition_counts = {element: 0 for element in required_elements}
    for element in required_elements:
        options = candidates.loc[
            (candidates["m_element"] == element)
            & ~candidates["candidate_id"].isin(selected)
        ]
        choice = rank(options, uncovered).iloc[0]
        selected.append(str(choice["candidate_id"]))
        selected_clusters.add(str(choice["structure_matcher_cluster"]))
        selected_composition_counts[element] += 1
        uncovered -= set(str(choice["represented_methods"]).split("|"))

    while uncovered and len(selected) < maximum_candidates:
        options = candidates.loc[~candidates["candidate_id"].isin(selected)]
        if composition_quotas:
            within_quota = options.loc[
                options.apply(
                    lambda row: selected_composition_counts.get(
                        str(row["m_element"]), 0
                    )
                    < composition_quotas.get(str(row["m_element"]), 0),
                    axis=1,
                )
            ]
            if not within_quota.empty and (
                rank(within_quota, uncovered).iloc[0]["_new_coverage"] > 0
            ):
                options = within_quota
        ranked = rank(options, uncovered)
        if ranked.empty or int(ranked.iloc[0]["_new_coverage"]) == 0:
            break
        choice = ranked.iloc[0]
        selected.append(str(choice["candidate_id"]))
        selected_clusters.add(str(choice["structure_matcher_cluster"]))
        element = str(choice["m_element"])
        selected_composition_counts[element] = (
            selected_composition_counts.get(element, 0) + 1
        )
        uncovered -= set(str(choice["represented_methods"]).split("|"))

    if uncovered:
        raise ValueError(
            "cannot represent every baseline within the candidate cap: "
            f"{sorted(uncovered)}"
        )

    while len(selected) < min(maximum_candidates, len(candidates)):
        options = candidates.loc[~candidates["candidate_id"].isin(selected)]
        if options.empty:
            break
        if composition_quotas:
            deficits = {
                element: composition_quotas.get(element, 0)
                - selected_composition_counts.get(element, 0)
                for element in required_elements
            }
            positive = [element for element, value in deficits.items() if value > 0]
            if positive:
                target = sorted(
                    positive,
                    key=lambda element: (-deficits[element], element),
                )[0]
                matched = options.loc[options["m_element"] == target]
                if not matched.empty:
                    options = matched
        choice = rank(options, set()).iloc[0]
        selected.append(str(choice["candidate_id"]))
        selected_clusters.add(str(choice["structure_matcher_cluster"]))
        element = str(choice["m_element"])
        selected_composition_counts[element] = (
            selected_composition_counts.get(element, 0) + 1
        )

    result = candidates.set_index("candidate_id").loc[selected].reset_index()
    result["freeze_rank"] = np.arange(1, len(result) + 1)
    if composition_quotas:
        result["target_composition_quota"] = result["m_element"].map(
            composition_quotas
        )
        actual = result["m_element"].value_counts()
        result["composition_quota_deviation"] = result["m_element"].map(
            lambda element: int(actual.get(element, 0))
            - int(composition_quotas.get(element, 0))
        )
    return result


def _stable_rank(
    frame: pd.DataFrame,
    column: str,
    *,
    ascending: bool = False,
) -> pd.DataFrame:
    return frame.sort_values(
        [column, "candidate_id"],
        ascending=[ascending, True],
        kind="mergesort",
    )


def _numeric_embedding(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> np.ndarray:
    encoded = pd.get_dummies(
        frame.loc[:, feature_columns],
        dummy_na=True,
        dtype=float,
    )
    encoded = encoded.replace([np.inf, -np.inf], np.nan)
    encoded = encoded.fillna(encoded.median(numeric_only=True)).fillna(0.0)
    values = encoded.to_numpy(dtype=float)
    scale = values.std(axis=0, ddof=0)
    scale[scale == 0] = 1.0
    return (values - values.mean(axis=0)) / scale


def _farthest_first(
    frame: pd.DataFrame,
    embedding: np.ndarray,
    *,
    batch_size: int,
    utility_column: str,
    frontier_multiplier: int = 3,
) -> list[str]:
    frontier_size = min(len(frame), max(batch_size, batch_size * frontier_multiplier))
    order = _stable_rank(frame, utility_column).index[:frontier_size].to_numpy()
    frontier = frame.loc[order]
    vectors = embedding[order]
    first_position = 0
    selected_positions = [first_position]
    while len(selected_positions) < min(batch_size, len(frontier)):
        chosen = vectors[selected_positions]
        distances = np.linalg.norm(
            vectors[:, None, :] - chosen[None, :, :], axis=2
        ).min(axis=1)
        distances[selected_positions] = -np.inf
        maximum = np.nanmax(distances)
        tied = np.flatnonzero(np.isclose(distances, maximum))
        if len(tied) > 1:
            tied_ids = frontier.iloc[tied]["candidate_id"].astype(str).to_numpy()
            choice = int(tied[np.argmin(tied_ids)])
        else:
            choice = int(tied[0])
        selected_positions.append(choice)
    return frontier.iloc[selected_positions]["candidate_id"].astype(str).tolist()


def _composition_probabilities(
    feature_history: pd.DataFrame,
    labels: np.ndarray,
    prospective: pd.DataFrame,
) -> np.ndarray:
    rows = []
    elements = feature_history["m_element"].astype(str)
    for element in prospective["m_element"].astype(str):
        local = labels[elements.to_numpy() == element]
        rows.append(float((local.sum() + 1.0) / (len(local) + 2.0)))
    return np.asarray(rows)


def _select_batch(
    frame: pd.DataFrame,
    embedding: np.ndarray,
    *,
    method: str,
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[list[str], bool, float, float]:
    available = frame.copy().reset_index(drop=True)
    available["current_utility"] = (
        0.70 * available["current_p_eval"]
        + 0.30 * available["p_interval_hit"]
    )
    available["joint_qualified_probability"] = (
        available["current_p_eval"] * available["p_interval_hit"]
    )
    direct = _stable_rank(available, "current_utility")
    top = direct.head(batch_size)
    margin = float(
        top["current_utility"].max() - top["current_utility"].min()
    )
    concentration = float(
        top["m_element"].value_counts(normalize=True).max()
    )

    if method == "random_sampling":
        chosen = rng.choice(
            available["candidate_id"].to_numpy(),
            size=min(batch_size, len(available)),
            replace=False,
        ).tolist()
        return chosen, False, margin, concentration
    if method == "predicted_target_greedy":
        chosen = _stable_rank(available, "p_interval_alignn").head(batch_size)
        return chosen["candidate_id"].tolist(), False, margin, concentration
    if method == "joint_qualified_greedy":
        chosen = _stable_rank(
            available, "joint_qualified_probability"
        ).head(batch_size)
        return chosen["candidate_id"].tolist(), False, margin, concentration
    if method == "dft_evaluable_greedy":
        chosen = _stable_rank(available, "current_p_eval").head(batch_size)
        return chosen["candidate_id"].tolist(), False, margin, concentration
    if method == "mc_uncertainty_only":
        chosen = _stable_rank(available, "current_uncertainty").head(batch_size)
        return chosen["candidate_id"].tolist(), False, margin, concentration
    if method == "composition_only":
        chosen = _stable_rank(available, "composition_probability").head(
            batch_size
        )
        return chosen["candidate_id"].tolist(), False, margin, concentration
    if method == "mlip_energy_greedy":
        chosen = _stable_rank(available, "p_interval_mlip").head(batch_size)
        return chosen["candidate_id"].tolist(), False, margin, concentration
    if method == "gradient_norm_hybrid":
        gradient = (
            available["current_p_eval"]
            * (1.0 - available["current_p_eval"])
            * np.linalg.norm(embedding, axis=1)
        )
        available["gradient_norm_proxy"] = gradient
        half = batch_size // 2
        first = _stable_rank(available, "gradient_norm_proxy").head(half)
        remaining = available.loc[
            ~available["candidate_id"].isin(first["candidate_id"])
        ]
        second = _stable_rank(remaining, "p_interval_hit").head(batch_size - half)
        return (
            [*first["candidate_id"].tolist(), *second["candidate_id"].tolist()],
            False,
            margin,
            concentration,
        )
    if method == "explore_core_set":
        return (
            _farthest_first(
                available,
                embedding,
                batch_size=batch_size,
                utility_column="current_utility",
            ),
            True,
            margin,
            concentration,
        )

    correction = gate_uses_correction(
        method,
        margin=margin,
        concentration=concentration,
    )
    if correction:
        chosen = _farthest_first(
            available,
            embedding,
            batch_size=batch_size,
            utility_column="current_utility",
        )
    else:
        chosen = direct.head(batch_size)["candidate_id"].tolist()
    return chosen, correction, margin, concentration


def _pseudo_oracle_sha256(oracle: pd.DataFrame) -> str:
    canonical = oracle.sort_values("candidate_id", kind="mergesort").to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.12g",
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def run_paired_replay(
    prospective: pd.DataFrame,
    history_x: pd.DataFrame,
    history_y: np.ndarray,
    *,
    model_feature_columns: tuple[str, ...],
    paired_seeds: tuple[int, ...],
    methods: tuple[str, ...],
    batch_size: int,
    query_budget: int,
    checkpoints: tuple[int, ...],
    classifier_model_name: str,
    bootstrap_draws: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run an active, paired pseudo-label replay with per-batch proxy retraining."""
    missing_methods = set(methods) - set(ALL_METHODS)
    if missing_methods:
        raise ValueError(f"unknown replay methods: {sorted(missing_methods)}")
    if query_budget > len(prospective):
        raise ValueError("query budget exceeds prospective pool")
    fixed_parameter = select_binary_parameter(
        history_x.loc[:, model_feature_columns],
        history_y,
        model_name=classifier_model_name,
        random_seed=8128,
    )
    results: list[dict[str, object]] = []
    selections: list[dict[str, object]] = []
    interval = (-2.3, -1.5)

    for seed in paired_seeds:
        oracle = make_paired_pseudo_oracle(prospective, seed=seed)
        oracle_hash = _pseudo_oracle_sha256(oracle)
        oracle_map = oracle.set_index("candidate_id")
        for method_index, method in enumerate(methods):
            selected_ids: list[str] = []
            selected_rows: list[dict[str, object]] = []
            train_x = history_x.loc[:, model_feature_columns].copy().reset_index(
                drop=True
            )
            train_y = np.asarray(history_y, dtype=int).copy()
            correction_rounds = 0
            rng = np.random.default_rng(seed * 1009 + method_index)

            while len(selected_ids) < query_budget:
                available = prospective.loc[
                    ~prospective["candidate_id"].isin(selected_ids)
                ].copy()
                batch_n = min(batch_size, query_budget - len(selected_ids))
                proxy_model_seed = seed * 100000 + len(selected_ids)
                predictions = bootstrap_binary_predictions(
                    train_x,
                    train_y,
                    available.loc[:, model_feature_columns],
                    model_name=classifier_model_name,
                    random_seed=proxy_model_seed,
                    draws=bootstrap_draws,
                    fixed_parameter=fixed_parameter,
                )
                available["current_p_eval"] = predictions.mean
                available["current_uncertainty"] = predictions.standard_deviation
                available["composition_probability"] = _composition_probabilities(
                    train_x,
                    train_y,
                    available,
                )
                embedding = _numeric_embedding(available, model_feature_columns)
                chosen, correction, margin, concentration = _select_batch(
                    available,
                    embedding,
                    method=method,
                    batch_size=batch_n,
                    rng=rng,
                )
                correction_rounds += int(correction)
                batch = available.set_index("candidate_id").loc[chosen].reset_index()
                for order, candidate in enumerate(batch.itertuples(index=False), 1):
                    candidate_id = str(candidate.candidate_id)
                    oracle_row = oracle_map.loc[candidate_id]
                    query = len(selected_ids) + order
                    record = {
                        "method": method,
                        "seed": seed,
                        "query": query,
                        "round": int(np.ceil(query / batch_size)),
                        "candidate_id": candidate_id,
                        "m_element": candidate.m_element,
                        "structure_matcher_cluster": candidate.structure_matcher_cluster,
                        "atom_count": candidate.atom_count,
                        "space_group_number": candidate.space_group_number,
                        "predicted_p_dft_evaluable": float(
                            candidate.current_p_eval
                        ),
                        "predicted_evaluability_uncertainty": float(
                            candidate.current_uncertainty
                        ),
                        "predicted_p_interval_hit": float(candidate.p_interval_hit),
                        "pseudo_dft_evaluable": int(
                            oracle_row["pseudo_dft_evaluable"]
                        ),
                        "pseudo_dft_energy_eV_atom": float(
                            oracle_row["pseudo_dft_energy_eV_atom"]
                        ),
                        "correction_route": bool(correction),
                        "round_margin": margin,
                        "round_composition_concentration": concentration,
                        "pseudo_oracle_sha256": oracle_hash,
                        "proxy_model_seed": int(proxy_model_seed),
                        "evidence_tier": "retrospective_ML_assisted_simulation",
                    }
                    selected_rows.append(record)
                selected_ids.extend(chosen)
                queried_x = batch.loc[:, model_feature_columns]
                queried_y = oracle_map.loc[
                    chosen, "pseudo_dft_evaluable"
                ].to_numpy(dtype=int)
                train_x = pd.concat([train_x, queried_x], ignore_index=True)
                train_y = np.concatenate([train_y, queried_y])

            method_selection = pd.DataFrame(selected_rows)
            selections.extend(selected_rows)
            for checkpoint in checkpoints:
                subset = method_selection.loc[
                    method_selection["query"] <= checkpoint
                ]
                real_band = subset["pseudo_dft_energy_eV_atom"].between(
                    interval[0], interval[1], inclusive="both"
                )
                results.append(
                    {
                        "method": method,
                        "seed": seed,
                        "checkpoint": checkpoint,
                        "estimated_DFT_evaluable_count": float(
                            subset["predicted_p_dft_evaluable"].sum()
                        ),
                        "simulated_DFT_evaluable_count": int(
                            subset["pseudo_dft_evaluable"].sum()
                        ),
                        "estimated_interval_hit_count": float(
                            (
                                subset["predicted_p_dft_evaluable"]
                                * subset["predicted_p_interval_hit"]
                            ).sum()
                        ),
                        "simulated_interval_hit_count": int(
                            (
                                subset["pseudo_dft_evaluable"].eq(1)
                                & real_band
                            ).sum()
                        ),
                        "unique_structure_clusters": int(
                            subset["structure_matcher_cluster"].nunique()
                        ),
                        "unique_compositions": int(subset["m_element"].nunique()),
                        "correction_rounds": int(
                            subset.loc[
                                subset["correction_route"], "round"
                            ].nunique()
                        ),
                        "pseudo_oracle_sha256": oracle_hash,
                        "evidence_tier": "retrospective_ML_assisted_simulation",
                    }
                )
    return pd.DataFrame(results), pd.DataFrame(selections)
