from __future__ import annotations

import pandas as pd
import pytest

from analysis.postprocess_formal_results import (
    build_paired_comparisons,
    formation_energy_per_atom,
    select_lower_energy_configurations,
    validated_toten,
    validate_formal_gpu_grid,
)


def _formal_grid() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for seed in range(15, 25):
        for method in (
            "interval_hit_greedy",
            "always_da_tpp",
            "margin_only_gate",
            "group_only_gate",
            "energy_gated_da_tpp",
        ):
            rows.append(
                {
                    "formal_stage": "li_m_o_ablation",
                    "dataset": "limo",
                    "method": method,
                    "group_key": "element_system_current",
                    "seed": seed,
                    "K": 30,
                    "AUTC": seed / 100 + len(method) / 10_000,
                }
            )
        for method, group_key in (
            ("interval_hit_greedy", "element_system_current"),
            ("always_da_tpp", "element_system_current"),
            ("energy_gated_da_tpp", "element_system_current"),
            ("energy_gated_da_tpp", "coelement_block_multiset"),
            ("energy_gated_da_tpp", "coelement_iupac_group_set"),
        ):
            rows.append(
                {
                    "formal_stage": "mn_group_key",
                    "dataset": "mnoxide",
                    "method": method,
                    "group_key": group_key,
                    "seed": seed,
                    "K": 30,
                    "AUTC": seed / 100 + len(group_key) / 10_000,
                }
            )
    for seed in range(25, 30):
        for k in (3, 10, 30):
            for method in ("interval_hit_greedy", "energy_gated_da_tpp"):
                rows.append(
                    {
                        "formal_stage": "mc_dropout_sensitivity",
                        "dataset": "limo",
                        "method": method,
                        "group_key": "element_system_current",
                        "seed": seed,
                        "K": k,
                        "AUTC": seed / 100 + k / 10_000 + len(method) / 100_000,
                    }
                )
    return pd.DataFrame(rows)


def test_validate_formal_gpu_grid_accepts_only_the_frozen_130_job_grid():
    grid = _formal_grid()

    validate_formal_gpu_grid(grid)

    with pytest.raises(ValueError, match="formal GPU grid mismatch"):
        validate_formal_gpu_grid(grid.iloc[:-1])


def test_build_paired_comparisons_pairs_by_seed_and_preserves_exact_settings():
    rows = []
    for seed, greedy, full in ((15, 0.70, 0.72), (16, 0.75, 0.74), (17, 0.80, 0.83)):
        rows.extend(
            [
                {
                    "formal_stage": "li_m_o_ablation",
                    "dataset": "limo",
                    "method": "interval_hit_greedy",
                    "group_key": "element_system_current",
                    "seed": seed,
                    "K": 30,
                    "AUTC": greedy,
                },
                {
                    "formal_stage": "li_m_o_ablation",
                    "dataset": "limo",
                    "method": "energy_gated_da_tpp",
                    "group_key": "element_system_current",
                    "seed": seed,
                    "K": 30,
                    "AUTC": full,
                },
            ]
        )

    differences, statistics = build_paired_comparisons(
        pd.DataFrame(rows), bootstrap_samples=100_000, bootstrap_seed=20260719
    )

    assert differences["paired_AUTC_difference"].tolist() == pytest.approx([0.02, -0.01, 0.03])
    result = statistics["li_m_o_ablation:limo:energy_gated_da_tpp:element_system_current:K30"]
    assert result["bootstrap_samples"] == 100_000
    assert result["bootstrap_seed"] == 20260719
    assert result["wilcoxon"]["zero_method"] == "wilcox"
    assert result["wilcoxon"]["correction"] is False
    assert result["wilcoxon"]["alternative"] == "two-sided"
    assert result["wilcoxon"]["method"] == "exact"
    assert result["effect_size_dz"] == pytest.approx(0.6405126152203481)


def test_formation_energy_uses_raw_total_energy_and_per_atom_references():
    value = formation_energy_per_atom(
        total_energy_eV=-49.0,
        composition={"Li": 1, "Cr": 2, "O": 4},
        references_eV_per_atom={"Li": -2.0, "Cr": -6.0, "O": -5.0},
    )

    assert value == pytest.approx((-49.0 - (-34.0)) / 7.0)
    with pytest.raises(ValueError, match="missing elemental references"):
        formation_energy_per_atom(-49.0, {"Li": 1, "Mn": 2, "O": 4}, {"Li": -2.0, "O": -5.0})


def test_lower_energy_selection_is_limited_to_two_tested_initializations():
    frame = pd.DataFrame(
        [
            {"candidate_id": "C044", "functional": "GGA+U", "magnetic_initialization": "state_fm", "final_total_energy_eV": -47.85},
            {"candidate_id": "C044", "functional": "GGA+U", "magnetic_initialization": "state_afm", "final_total_energy_eV": -47.78},
        ]
    )

    selected = select_lower_energy_configurations(frame)

    assert selected.loc[selected["selected_lower_energy_among_two_tested"], "magnetic_initialization"].item() == "state_fm"
    assert set(selected["selection_scope"]) == {"lower-energy configuration among the two tested initializations"}

    with pytest.raises(ValueError, match="exactly two tested magnetic initializations"):
        select_lower_energy_configurations(frame.iloc[:1])


def test_validated_toten_uses_matching_free_energy_channel_not_sigma_zero_energy():
    # VASP prints TOTEN (free energy) and energy(sigma->0) as distinct channels.
    # Pymatgen's final_energy uses the latter, so it is not the consistency check.
    value = validated_toten(
        outcar_toten_eV=-47.66013567,
        vasprun_free_energy_eV=-47.66013567,
    )

    assert value == pytest.approx(-47.66013567)
    with pytest.raises(ValueError, match="OUTCAR TOTEN and vasprun free energy differ"):
        validated_toten(-47.66013567, -47.6600)
