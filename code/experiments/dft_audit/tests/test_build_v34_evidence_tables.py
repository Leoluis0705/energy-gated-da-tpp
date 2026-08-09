import csv
import json
from pathlib import Path

from analysis.build_v34_evidence_tables import (
    _dft_verification_complete,
    _path,
    build_v34_tables,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_long_paths_use_breakable_latex_path_command() -> None:
    assert _path("D:/archive/checkpoint_formation_clean.pth.tar") == (
        r"\path{D:/archive/checkpoint_formation_clean.pth.tar}"
    )


def test_dft_verification_completion_requires_both_candidates() -> None:
    complete = "completed_frozen_protocol_relaxation_and_static"
    rows = [
        {"candidate_label": "C044", "verification_status": "archived_assessment"},
        {"candidate_label": "C120", "verification_status": complete},
        {"candidate_label": "C214", "verification_status": complete},
    ]
    assert _dft_verification_complete(rows) is True
    rows[-1]["verification_status"] = "verification_relaxation_pending"
    assert _dft_verification_complete(rows) is False


def test_tables_use_raw_analysis_outputs_and_not_v33_values(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    supplemental = tmp_path / "supplementary"
    output = tmp_path / "generated"

    _write_csv(
        bundle / "gpu" / "method_summary.csv",
        [
            {
                "formal_stage": "li_m_o_ablation",
                "dataset": "limo",
                "method": method,
                "group_key": "element_system_current",
                "K": 30,
                "n_seeds": 10,
                "AUTC_mean": mean,
                "AUTC_sample_sd": sd,
                "recovery_at_80_mean": r80,
                "recovery_at_80_sample_sd": 1.0,
                "recovery_at_160_mean": 50,
                "recovery_at_160_sample_sd": 2.0,
                "recovery_at_240_mean": 70,
                "recovery_at_240_sample_sd": 1.0,
                "recovery_at_320_mean": 78,
                "recovery_at_320_sample_sd": 0.0,
                "direct_rounds_mean": direct,
                "correction_rounds_mean": correction,
                "effective_replacements_mean": replacements,
                "mean_unique_groups_per_batch_mean": 3.2,
                "repetition_rate_mean": 0.8,
            }
            for method, mean, sd, r80, direct, correction, replacements in [
                ("interval_hit_greedy", 0.784679, 0.010928, 25.8, 40, 0, 0),
                ("always_da_tpp", 0.792660, 0.012619, 27.5, 0, 40, 217.8),
                ("margin_only_gate", 0.784679, 0.010928, 25.8, 39.9, 0.1, 0.9),
                ("group_only_gate", 0.794840, 0.010300, 28.4, 4.7, 35.3, 201.7),
                ("energy_gated_da_tpp", 0.794840, 0.010300, 28.4, 4.7, 35.3, 201.7),
            ]
        ],
    )
    paired = {
        "comparisons": {
            "li_m_o_ablation:limo:energy_gated_da_tpp:element_system_current:K30": {
                "paired_mean": 0.0101602564,
                "paired_sd": 0.0129842201,
                "bootstrap_ci_95_percentile": [0.0028205128, 0.0180448718],
                "effect_size_dz": 0.782508,
                "wilcoxon": {"pvalue": 0.0546875, "status": "computed"},
            },
            "li_m_o_ablation:limo:always_da_tpp:element_system_current:K30": {
                "paired_mean": 0.0079807692,
                "paired_sd": 0.0125691618,
                "bootstrap_ci_95_percentile": [0.0009935897, 0.0156730769],
                "effect_size_dz": 0.634948,
                "wilcoxon": {"pvalue": 0.130859375, "status": "computed"},
            },
            "li_m_o_ablation:limo:margin_only_gate:element_system_current:K30": {
                "paired_mean": 0.0,
                "paired_sd": 0.0,
                "bootstrap_ci_95_percentile": [0.0, 0.0],
                "effect_size_dz": 0.0,
                "wilcoxon": {"pvalue": None, "status": "not_applicable_all_zero"},
            },
            "li_m_o_ablation:limo:group_only_gate:element_system_current:K30": {
                "paired_mean": 0.0101602564,
                "paired_sd": 0.0129842201,
                "bootstrap_ci_95_percentile": [0.0028205128, 0.0180448718],
                "effect_size_dz": 0.782508,
                "wilcoxon": {"pvalue": 0.0546875, "status": "computed"},
            },
        },
        "environment": {"bootstrap_samples": 100000, "bootstrap_seed": 20260719},
    }
    (bundle / "gpu" / "paired_statistics.json").write_text(
        json.dumps(paired), encoding="utf-8"
    )
    _write_csv(
        bundle / "gpu" / "mn_group_key_sensitivity_summary.csv",
        [
            {
                "group_key": "element_system_current",
                "display_label": "Element system",
                "group_count": 614,
                "singleton_group_count": 588,
                "singleton_group_fraction": 0.95765,
                "maximum_group_size": 2,
                "formal_top_b_concentration_max": 0.125,
                "minimum_margin_score": 1.0967,
                "correction_rounds_total": 0,
                "effective_replacements_total": 0,
                "AUTC_mean": 0.432207,
                "AUTC_sample_sd": 0.022813,
            }
        ],
    )
    _write_csv(
        bundle / "gpu" / "mc_dropout_summary.csv",
        [
            {
                "mc_passes": k,
                "median_uncertainty_spearman_vs_k30": rho,
                "median_top_b_overlap_vs_k30": overlap,
                "gate_flip_rate_vs_k30": flips,
                "mean_absolute_AUTC_difference_vs_k30": delta,
                "mean_runtime_seconds": runtime,
                "median_runtime_ratio_vs_k30": ratio,
            }
            for k, rho, overlap, flips, delta, runtime, ratio in [
                (3, 0.1749, 0.0625, 0.255, 0.01548, 944.2, 0.9673),
                (10, 0.2917, 0.125, 0.2, 0.01369, 953.7, 0.9762),
                (30, 1.0, 1.0, 0.0, 0.0, 970.7, 1.0),
            ]
        ],
    )
    _write_csv(
        bundle / "dft" / "main_text_table7_comparison.csv",
        [
            {
                "candidate_label": "C120",
                "candidate_id": "candidate-120",
                "selected_magnetic_initialization": "state_fm",
                "recomputed_formation_energy_eV_per_atom": -2.25603279,
                "v33_Table7_printed_formation_energy_eV_per_atom": -9.9999,
                "recomputed_minus_v33_printed_eV_per_atom": 7.74386721,
            }
        ],
    )
    _write_csv(
        bundle / "dft" / "dft_candidate_manifest.csv",
        [
            {
                "candidate_id": "candidate-120",
                "formula": "LiCr2O4",
                "pilot_or_new": "new",
                "selection_rule": "boundary",
                "freeze_timestamp": "",
                "timestamp_source": "unavailable",
                "result_known_at_freeze": "false",
                "Gate_round": 3,
                "Greedy_round": 9,
                "DFT_status": "static_finished",
                "failure_reason": "",
                "main_text_selected": "true",
                "sha256": "abc123",
            }
        ],
    )
    _write_csv(
        bundle / "dft" / "elemental_references.csv",
        [{
            "element": "Cr", "functional": "GGA+U", "structure": "Im-3m (229)",
            "magnetic_setup": "ISPIN=2; MAGMOM=2*5.0", "paw_label": "PAW_PBE Cr_pv",
            "Ueff_eV": 3.7, "kpoints_mesh": "15x15x15", "energy_per_atom_eV": -5.82,
            "electronic_converged": "True", "raw_output_path": "D:/raw/OUTCAR",
            "raw_output_sha256": "a" * 64, "reference_risk": "",
        }],
    )
    _write_csv(
        bundle / "dft" / "structure_metrics.csv",
        [{
            "candidate_id": "job_120_Cr_example", "functional": "GGA+U",
            "magnetic_initialization": "state_fm",
            "historical_selected_configuration_initial_volume_A3": 70,
            "historical_selected_configuration_final_volume_A3": 71,
            "historical_selected_configuration_relative_volume_change_percent": 1.4,
            "minimum_interatomic_distance_A": 1.9, "minimum_M_O_distance_A": 1.9,
            "historical_selected_configuration_relaxation_Fmax_eV_A": 0.04,
            "Fmax_eV_A_static_diagnostic": 0.05, "final_space_group": "P1 (1)",
        }],
    )
    _write_csv(
        bundle / "dft" / "verification_relaxation_metrics.csv",
        [{
            "candidate_id": "C120", "magnetic_initialization": "state_fm",
            "initial_a_A": 2.93, "initial_b_A": 5.03, "initial_c_A": 5.09,
            "initial_alpha_deg": 104.1, "initial_beta_deg": 90.5,
            "initial_gamma_deg": 105.7,
            "final_a_A": 2.94, "final_b_A": 5.04, "final_c_A": 5.10,
            "final_alpha_deg": 104.0, "final_beta_deg": 90.6,
            "final_gamma_deg": 105.7,
            "initial_volume_A3": 70.0, "final_volume_A3": 70.1,
            "relative_volume_change_percent": 0.14,
            "maximum_internal_displacement_A": 0.003,
            "minimum_interatomic_distance_A": 1.80,
            "minimum_M_O_distance_A": 1.80, "Fmax_eV_A": 0.03,
            "initial_space_group": "P1 (1)", "final_space_group": "P1 (1)",
        }],
    )
    _write_csv(
        bundle / "dft" / "verification_static_metrics.csv",
        [{
            "candidate_id": "C120", "magnetic_initialization": "state_fm",
            "electronic_converged": True, "final_total_energy_eV": -49.11,
            "Fmax_eV_A_static_diagnostic": 0.02,
            "final_total_magnetic_moment": 5.0, "kpoints_mesh": "15x9x9",
            "final_space_group": "P1 (1)",
        }],
    )
    _write_csv(
        bundle / "dft" / "selected_candidate_comparison.csv",
        [{
            "candidate_id": "C120", "historical_selected_initialization": "state_fm",
            "new_selected_initialization": "state_fm",
            "historical_selected_formation_energy_eV_per_atom": -2.2560,
            "new_selected_formation_energy_eV_per_atom": -2.2561,
            "selected_formation_energy_shift_eV_per_atom": -0.0001,
        }],
    )
    _write_csv(
        supplemental / "tables" / "table_s_parameter_calibration.csv",
        [{"stage": "weight_seeds0_4", "cohort": "seeds0_4", "selection_rank": 1, "config_id": "gamma_0p05", "M0": 0.75, "G0": 0.5, "alpha": 0.1, "beta": 0.2, "gamma": 0.05}],
    )
    _write_csv(
        supplemental / "tables" / "table_s_checkpoint_provenance.csv",
        [{"checkpoint_path": "checkpoint.pth.tar", "sha256": "def456", "normalizer_mean": -2.5, "normalizer_scale": 0.79, "exact_cif_hash_overlap_count": 0}],
    )

    manifest = build_v34_tables(bundle, supplemental, output)

    macros = (output / "formal_metrics_macros.tex").read_text(encoding="utf-8")
    assert "0.794840" in macros
    assert "0.0546875" in macros
    assert "-9.9999" not in macros
    assert "PENDING VERIFICATION" in (output / "table7_dft_provisional.tex").read_text(encoding="utf-8")
    dft_manifest = (output / "table_s_dft_manifest.tex").read_text(encoding="utf-8")
    assert "candidate-120" in dft_manifest
    assert "unavailable" in dft_manifest
    assert (output / "table_s_dft_manifest_status.tex").exists()
    assert (output / "table_s_dft_manifest_provenance.tex").exists()
    assert max(line.count("&") for line in dft_manifest.splitlines()) <= 7
    for name in [
        "table_s_checkpoint_provenance.tex",
        "table_s_dft_manifest_status.tex",
        "table_s_dft_manifest_provenance.tex",
    ]:
        assert "p{" not in (output / name).read_text(encoding="utf-8")
    assert (output / "table_s_elemental_reference_provenance.tex").exists()
    assert (output / "table_s_structure_geometry_forces.tex").exists()
    assert (output / "table_s_verification_lattice.tex").exists()
    assert (output / "table_s_verification_angles.tex").exists()
    verification_geometry = (output / "table_s_verification_geometry.tex").read_text(
        encoding="utf-8"
    )
    assert r"Space group $i\rightarrow f$" in verification_geometry
    assert "\r" not in verification_geometry
    assert (output / "table_s_verification_statics.tex").exists()
    assert (output / "table_s_verification_energy_shift.tex").exists()
    assert all("v33_tables4_6_comparison.csv" not in item["source"] for item in manifest)
