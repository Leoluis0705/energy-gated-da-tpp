"""Build the frozen Cr/Mn/Mg pool and auditable historical DFT labels."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


INCLUDED_ELEMENTS = ("Cr", "Mn", "Mg")
EXPECTED_COUNTS = {"Cr": 114, "Mn": 127, "Mg": 34}
MLIP_MODELS = {"CHGNet": "chgnet", "MACE-MP": "mace"}

MLIP_NUMERIC_COLUMNS = (
    "initial_energy_eV_atom",
    "final_energy_eV_atom",
    "max_force_eV_A",
    "stress_GPa",
    "initial_volume_A3",
    "final_volume_A3",
    "volume_change_fraction",
    "min_interatomic_distance_A",
    "rms_displacement_A",
    "final_space_group_number",
    "merged_candidate_count",
    "wall_time_seconds",
)

MODEL_FEATURE_COLUMNS = (
    "m_element",
    "atom_count",
    "space_group_number",
    "structure_matcher_cluster_size",
    "min_interatomic_distance",
    "volume_per_atom",
    "chgnet_initial_energy_eV_atom",
    "chgnet_final_energy_eV_atom",
    "chgnet_max_force_eV_A",
    "chgnet_stress_GPa",
    "chgnet_volume_change_fraction",
    "chgnet_min_interatomic_distance_A",
    "chgnet_rms_displacement_A",
    "chgnet_final_space_group_number",
    "mace_initial_energy_eV_atom",
    "mace_final_energy_eV_atom",
    "mace_max_force_eV_A",
    "mace_stress_GPa",
    "mace_volume_change_fraction",
    "mace_min_interatomic_distance_A",
    "mace_rms_displacement_A",
    "mace_final_space_group_number",
    "mlip_energy_disagreement_eV_atom",
    "mlip_rms_displacement_disagreement_A",
    "mlip_volume_change_disagreement_fraction",
)


def _as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def _atom_count(formula: str) -> float:
    matches = re.findall(r"[A-Z][a-z]?\s*([0-9]*\.?[0-9]*)", str(formula))
    if not matches:
        return np.nan
    values = [1.0 if value == "" else float(value) for value in matches]
    return float(sum(values))


def _select_protocol_dft_energies(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame.loc[_as_bool(frame["selected_for_formation_energy"])].copy()
    selected["formation_energy_eV_per_atom"] = pd.to_numeric(
        selected["formation_energy_eV_per_atom"], errors="coerce"
    )
    selected = selected.dropna(subset=["formation_energy_eV_per_atom"])
    priority = {"GGA+U": 0, "PBE+U": 0, "PBE": 1}
    selected["_priority"] = selected["functional"].map(priority).fillna(99)
    return (
        selected.sort_values(
            ["candidate_id", "_priority", "functional", "job_id"],
            kind="mergesort",
        )
        .drop_duplicates("candidate_id", keep="first")
        .drop(columns="_priority")
        .reset_index(drop=True)
    )


def build_historical_binary_labels(
    manifest_path: str | Path,
    formation_energy_path: str | Path,
) -> pd.DataFrame:
    """Apply the frozen strict evaluability rule to all historical attempts."""
    manifest = pd.read_csv(manifest_path)
    energies = _select_protocol_dft_energies(pd.read_csv(formation_energy_path))
    energy_map = energies.set_index("candidate_id")
    quantitative = set(energies["candidate_id"])

    rows: list[dict[str, object]] = []
    for row in manifest.itertuples(index=False):
        static_finished = str(row.DFT_status) == "static_finished"
        reproducible = row.candidate_id in quantitative
        evaluable = int(static_finished and reproducible)
        if evaluable:
            reason = "strict_DFT_chain_and_formation_energy_verified"
        elif static_finished:
            reason = "formation_energy_not_reproducible"
        elif str(row.DFT_status) == "failed_static_electronic_nonconvergence":
            reason = "static_electronic_nonconvergence"
        elif str(row.failure_reason).strip():
            reason = str(row.failure_reason).strip()
        else:
            reason = str(row.DFT_status)
        energy = (
            float(energy_map.loc[row.candidate_id, "formation_energy_eV_per_atom"])
            if reproducible
            else np.nan
        )
        functional = (
            str(energy_map.loc[row.candidate_id, "functional"])
            if reproducible
            else ""
        )
        rows.append(
            {
                "candidate_id": row.candidate_id,
                "formula": row.formula,
                "m_element": re.sub(r"^Li|2O4$", "", str(row.formula)),
                "pilot_or_new": row.pilot_or_new,
                "dft_evaluable": evaluable,
                "label_reason": reason,
                "dft_status": row.DFT_status,
                "relaxation_output_available": bool(
                    str(row.relaxation_output_available).lower() == "true"
                ),
                "static_output_available": bool(
                    str(row.static_output_available).lower() == "true"
                ),
                "formation_energy_reproducible": reproducible,
                "dft_formation_energy_eV_atom": energy,
                "dft_functional": functional,
                "source_manifest_path": str(Path(manifest_path)),
                "source_energy_path": str(Path(formation_energy_path)),
            }
        )
    result = pd.DataFrame(rows)
    if result["candidate_id"].duplicated().any():
        raise ValueError("historical DFT manifest contains duplicate candidates")
    return result.sort_values("candidate_id", kind="mergesort").reset_index(drop=True)


def _pivot_mlip(mlip: pd.DataFrame) -> pd.DataFrame:
    missing_models = set(MLIP_MODELS) - set(mlip["model_name"].dropna().unique())
    if missing_models:
        raise ValueError(f"missing frozen MLIP models: {sorted(missing_models)}")
    counts = mlip.groupby("candidate_id")["model_name"].nunique()
    if not counts.eq(2).all():
        bad = counts.loc[~counts.eq(2)].index.tolist()[:5]
        raise ValueError(f"candidates do not have two MLIP rows: {bad}")

    parts = []
    for model_name, prefix in MLIP_MODELS.items():
        part = mlip.loc[mlip["model_name"] == model_name].copy()
        keep = ["candidate_id", *MLIP_NUMERIC_COLUMNS, "relaxed_structure_cluster"]
        part = part[keep]
        for column in MLIP_NUMERIC_COLUMNS:
            part[column] = pd.to_numeric(part[column], errors="coerce")
        part = part.rename(
            columns={
                column: f"{prefix}_{column}" for column in MLIP_NUMERIC_COLUMNS
            }
            | {"relaxed_structure_cluster": f"{prefix}_relaxed_structure_cluster"}
        )
        parts.append(part)
    pivot = parts[0].merge(parts[1], on="candidate_id", validate="one_to_one")
    pivot["mlip_model_count"] = 2
    pivot["mlip_energy_disagreement_eV_atom"] = (
        pivot["chgnet_final_energy_eV_atom"] - pivot["mace_final_energy_eV_atom"]
    ).abs()
    pivot["mlip_rms_displacement_disagreement_A"] = (
        pivot["chgnet_rms_displacement_A"] - pivot["mace_rms_displacement_A"]
    ).abs()
    pivot["mlip_volume_change_disagreement_fraction"] = (
        pivot["chgnet_volume_change_fraction"]
        - pivot["mace_volume_change_fraction"]
    ).abs()
    return pivot


def build_three_system_pool(
    pool_path: str | Path,
    mlip_path: str | Path,
    historical_manifest_path: str | Path,
) -> pd.DataFrame:
    """Join frozen pool metadata to the complete two-MLIP result table."""
    pool = pd.read_csv(pool_path)
    pool = pool.loc[pool["m_element"].isin(INCLUDED_ELEMENTS)].copy()
    counts = pool["m_element"].value_counts().to_dict()
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"three-system pool counts changed: {counts}")
    if pool["candidate_id"].duplicated().any():
        raise ValueError("candidate IDs are not unique")

    manifest = pd.read_csv(historical_manifest_path)
    historical_ids = set(manifest["candidate_id"])
    mlip = _pivot_mlip(pd.read_csv(mlip_path))
    result = pool.merge(mlip, on="candidate_id", how="left", validate="one_to_one")
    if result["mlip_model_count"].isna().any():
        raise ValueError("one or more three-system candidates lack MLIP results")
    result["historical_dft"] = result["candidate_id"].isin(historical_ids)
    result["atom_count"] = result["formula"].map(_atom_count)
    result["prospective_eligible"] = ~result["historical_dft"]
    result.attrs["model_feature_columns"] = list(MODEL_FEATURE_COLUMNS)
    return result.sort_values("candidate_id", kind="mergesort").reset_index(drop=True)


def model_feature_frame(pool: pd.DataFrame) -> pd.DataFrame:
    missing = set(MODEL_FEATURE_COLUMNS) - set(pool.columns)
    if missing:
        raise ValueError(f"pool is missing model features: {sorted(missing)}")
    return pool.loc[:, MODEL_FEATURE_COLUMNS].copy()

