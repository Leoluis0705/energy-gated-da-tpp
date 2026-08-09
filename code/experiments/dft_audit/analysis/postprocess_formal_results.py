"""Independent post-compute analysis for the frozen formal GPU and DFT runs."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from analysis.recompute_statistics import paired_statistics


FORMAL_SEEDS = tuple(range(15, 25))
MC_SENSITIVITY_SEEDS = tuple(range(25, 30))
MC_PASSES = (3, 10, 30)

LI_METHODS = (
    "interval_hit_greedy",
    "always_da_tpp",
    "margin_only_gate",
    "group_only_gate",
    "energy_gated_da_tpp",
)
MN_METHOD_GROUPS = (
    ("interval_hit_greedy", "element_system_current"),
    ("always_da_tpp", "element_system_current"),
    ("energy_gated_da_tpp", "element_system_current"),
    ("energy_gated_da_tpp", "coelement_block_multiset"),
    ("energy_gated_da_tpp", "coelement_iupac_group_set"),
)


def _formal_gpu_keys() -> set[tuple[str, str, str, str, int, int]]:
    keys: set[tuple[str, str, str, str, int, int]] = set()
    for seed in FORMAL_SEEDS:
        keys.update(
            (
                "li_m_o_ablation",
                "limo",
                method,
                "element_system_current",
                seed,
                30,
            )
            for method in LI_METHODS
        )
        keys.update(
            ("mn_group_key", "mnoxide", method, group_key, seed, 30)
            for method, group_key in MN_METHOD_GROUPS
        )
    for seed in MC_SENSITIVITY_SEEDS:
        keys.update(
            (
                "mc_dropout_sensitivity",
                "limo",
                method,
                "element_system_current",
                seed,
                k,
            )
            for k in MC_PASSES
            for method in ("interval_hit_greedy", "energy_gated_da_tpp")
        )
    return keys


def validate_formal_gpu_grid(metrics: pd.DataFrame) -> None:
    """Require the exact frozen 130-trajectory evaluation grid."""

    columns = ["formal_stage", "dataset", "method", "group_key", "seed", "K"]
    missing_columns = sorted(set(columns).difference(metrics.columns))
    if missing_columns:
        raise ValueError(f"formal GPU grid is missing columns: {missing_columns}")
    actual = {
        (str(stage), str(dataset), str(method), str(group_key), int(seed), int(k))
        for stage, dataset, method, group_key, seed, k in metrics[columns].itertuples(
            index=False, name=None
        )
    }
    expected = _formal_gpu_keys()
    if len(metrics) != len(expected) or actual != expected:
        raise ValueError(
            "formal GPU grid mismatch; "
            f"rows={len(metrics)}, expected_rows={len(expected)}, "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _comparison_key(row: pd.Series) -> str:
    return ":".join(
        [
            str(row["formal_stage"]),
            str(row["dataset"]),
            str(row["method"]),
            str(row["group_key"]),
            f"K{int(row['K'])}",
        ]
    )


def build_paired_comparisons(
    metrics: pd.DataFrame,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    """Pair every non-Greedy configuration with its matched Greedy seed."""

    required = {
        "formal_stage",
        "dataset",
        "method",
        "group_key",
        "seed",
        "K",
        "AUTC",
    }
    missing = sorted(required.difference(metrics.columns))
    if missing:
        raise ValueError(f"metrics are missing columns: {missing}")

    output_rows: list[dict[str, object]] = []
    statistics: dict[str, dict[str, object]] = {}
    comparison_columns = ["formal_stage", "dataset", "method", "group_key", "K"]
    alternatives = metrics[metrics["method"] != "interval_hit_greedy"]
    for comparison, alternative in alternatives.groupby(comparison_columns, sort=True):
        stage, dataset, method, group_key, k = comparison
        baseline = metrics[
            (metrics["formal_stage"] == stage)
            & (metrics["dataset"] == dataset)
            & (metrics["method"] == "interval_hit_greedy")
            & (metrics["K"] == k)
        ]
        joined = alternative.merge(
            baseline[["seed", "AUTC"]],
            on="seed",
            suffixes=("_method", "_greedy"),
            validate="one_to_one",
        )
        if len(joined) != len(alternative):
            raise ValueError(f"incomplete matched Greedy pairing for {comparison}")
        key_row = pd.Series(
            {
                "formal_stage": stage,
                "dataset": dataset,
                "method": method,
                "group_key": group_key,
                "K": k,
            }
        )
        key = _comparison_key(key_row)
        differences = joined["AUTC_method"] - joined["AUTC_greedy"]
        statistics[key] = paired_statistics(
            differences.to_numpy(dtype=float),
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
        for row, difference in zip(
            joined.to_dict(orient="records"), differences, strict=True
        ):
            output_rows.append(
                {
                    "comparison_id": key,
                    "formal_stage": stage,
                    "dataset": dataset,
                    "method": method,
                    "group_key": group_key,
                    "K": int(k),
                    "seed": int(row["seed"]),
                    "method_AUTC": float(row["AUTC_method"]),
                    "Greedy_AUTC": float(row["AUTC_greedy"]),
                    "paired_AUTC_difference": float(difference),
                }
            )
    frame = pd.DataFrame(output_rows).sort_values(
        ["formal_stage", "dataset", "K", "method", "group_key", "seed"]
    )
    return frame.reset_index(drop=True), statistics


def formation_energy_per_atom(
    total_energy_eV: float,
    composition: Mapping[str, int | float],
    references_eV_per_atom: Mapping[str, float],
) -> float:
    """Calculate formation energy directly from raw total/reference energies."""

    counts = {str(element): float(count) for element, count in composition.items()}
    if not counts or any(count <= 0 for count in counts.values()):
        raise ValueError("composition counts must be positive")
    missing = sorted(set(counts).difference(references_eV_per_atom))
    if missing:
        raise ValueError(f"missing elemental references: {missing}")
    atom_count = sum(counts.values())
    reference_total = sum(
        count * float(references_eV_per_atom[element])
        for element, count in counts.items()
    )
    return (float(total_energy_eV) - reference_total) / atom_count


def validated_toten(
    outcar_toten_eV: float,
    vasprun_free_energy_eV: float,
    *,
    tolerance_eV: float = 5e-8,
) -> float:
    """Validate and return VASP TOTEN using the matching free-energy channel."""

    difference = abs(float(outcar_toten_eV) - float(vasprun_free_energy_eV))
    if difference > tolerance_eV:
        raise ValueError(
            "OUTCAR TOTEN and vasprun free energy differ; "
            f"absolute difference={difference:.12g} eV"
        )
    return float(outcar_toten_eV)


def select_lower_energy_configurations(frame: pd.DataFrame) -> pd.DataFrame:
    """Mark only the lower-energy member of each two-initialization comparison."""

    required = {
        "candidate_id",
        "functional",
        "magnetic_initialization",
        "final_total_energy_eV",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"magnetic comparison is missing columns: {missing}")
    result = frame.copy()
    result["selected_lower_energy_among_two_tested"] = False
    result["selection_scope"] = (
        "lower-energy configuration among the two tested initializations"
    )
    for key, group in result.groupby(["candidate_id", "functional"], sort=True):
        if len(group) != 2 or group["magnetic_initialization"].nunique() != 2:
            raise ValueError(
                f"{key} requires exactly two tested magnetic initializations"
            )
        energies = pd.to_numeric(group["final_total_energy_eV"], errors="raise")
        result.loc[energies.idxmin(), "selected_lower_energy_among_two_tested"] = True
    return result
