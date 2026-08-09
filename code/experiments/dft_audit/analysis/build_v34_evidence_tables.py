#!/usr/bin/env python3
"""Generate v34 manuscript/SI tables from audited CSV and JSON evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable


COMPLETED_DFT_VERIFICATION_STATUS = (
    "completed_frozen_protocol_relaxation_and_static"
)


def _dft_verification_complete(rows: list[dict[str, object]]) -> bool:
    indexed = {str(row.get("candidate_label")): row for row in rows}
    return all(
        str(indexed.get(candidate, {}).get("verification_status", ""))
        == COMPLETED_DFT_VERIFICATION_STATUS
        for candidate in ("C120", "C214")
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tex(value: object) -> str:
    text = "" if value is None else str(value)
    if not text.strip():
        return "unavailable"
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def _number(value: object, digits: int = 6) -> str:
    if value in (None, ""):
        return "--"
    return f"{float(value):.{digits}f}"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")


def _candidate_display(candidate_id: str) -> str:
    match = re.search(r"job_(\d{3})_", candidate_id)
    return f"C{match.group(1)}" if match else _tex(candidate_id)


def _seqsplit(value: object) -> str:
    text = "unavailable" if value in (None, "") else str(value)
    if re.fullmatch(r"[0-9a-fA-F]+", text):
        return rf"\seqsplit{{{text}}}"
    return _tex(text)


def _path(value: object) -> str:
    text = "unavailable" if value in (None, "") else str(value)
    escaped = text.replace("}", r"\}")
    return r"\path{" + escaped + "}"


def _parbox(width: str, value: str) -> str:
    return rf"\parbox[t]{{{width}}}{{\raggedright {value}}}"


def _tabular(
    columns: str,
    header: Iterable[str],
    rows: Iterable[Iterable[object]],
    *,
    longtable: bool = False,
) -> str:
    environment = "longtable" if longtable else "tabular"
    lines = [f"\\begin{{{environment}}}{{{columns}}}", r"\toprule"]
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")
    for row in rows:
        lines.append(" & ".join(str(item) for item in row) + r" \\")
    lines.extend([r"\bottomrule", f"\\end{{{environment}}}"])
    return "\n".join(lines)


def _comparison_key(method: str) -> str:
    return f"li_m_o_ablation:limo:{method}:element_system_current:K30"


def _wilcoxon_text(comparison: dict[str, object] | None) -> str:
    if comparison is None:
        return "--"
    wilcoxon = comparison.get("wilcoxon", {})
    if wilcoxon.get("status") != "computed":
        return "not defined"
    return _number(wilcoxon.get("pvalue"), 7)


def _build_main_tables(bundle: Path, output: Path) -> list[Path]:
    method_path = bundle / "gpu" / "method_summary.csv"
    paired_path = bundle / "gpu" / "paired_statistics.json"
    group_path = bundle / "gpu" / "mn_group_key_sensitivity_summary.csv"
    mc_path = bundle / "gpu" / "mc_dropout_summary.csv"
    dft_path = bundle / "dft" / "main_text_table7_comparison.csv"

    methods = {
        row["method"]: row
        for row in _read_csv(method_path)
        if row.get("formal_stage") == "li_m_o_ablation"
        and row.get("dataset") == "limo"
        and row.get("K") == "30"
    }
    paired_document = json.loads(paired_path.read_text(encoding="utf-8"))
    paired = paired_document["comparisons"]
    environment = paired_document["environment"]
    groups = _read_csv(group_path)
    mc_rows = _read_csv(mc_path)
    dft_rows = _read_csv(dft_path)

    full = methods["energy_gated_da_tpp"]
    greedy = methods["interval_hit_greedy"]
    full_pair = paired[_comparison_key("energy_gated_da_tpp")]
    macros = {
        "FormalK": full["K"],
        "FormalSeedCount": full["n_seeds"],
        "FullAUTCMean": _number(full["AUTC_mean"]),
        "FullAUTCSD": _number(full["AUTC_sample_sd"]),
        "GreedyAUTCMean": _number(greedy["AUTC_mean"]),
        "GreedyAUTCSD": _number(greedy["AUTC_sample_sd"]),
        "FullPairedMean": _number(full_pair["paired_mean"]),
        "FullPairedSD": _number(full_pair["paired_sd"]),
        "FullBootstrapLow": _number(full_pair["bootstrap_ci_95_percentile"][0]),
        "FullBootstrapHigh": _number(full_pair["bootstrap_ci_95_percentile"][1]),
        "FullDz": _number(full_pair["effect_size_dz"], 4),
        "FullWilcoxonP": _number(full_pair["wilcoxon"]["pvalue"], 7),
        "BootstrapSamples": str(environment["bootstrap_samples"]),
        "BootstrapSeed": str(environment["bootstrap_seed"]),
    }
    macro_lines = [
        f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in macros.items()
    ]
    macro_path = output / "formal_metrics_macros.tex"
    _write(macro_path, "\n".join(macro_lines))

    labels = {
        "interval_hit_greedy": "Interval-Hit Greedy",
        "always_da_tpp": "Always-DA-TPP",
        "margin_only_gate": "Margin-only Gate",
        "group_only_gate": "Group-only Gate",
        "energy_gated_da_tpp": "Full Energy-Gated DA-TPP",
    }
    order = list(labels)
    table4_rows = []
    for method in order:
        row = methods[method]
        comp = None if method == "interval_hit_greedy" else paired.get(_comparison_key(method))
        ci = "--" if comp is None else (
            f"[{_number(comp['bootstrap_ci_95_percentile'][0])}, "
            f"{_number(comp['bootstrap_ci_95_percentile'][1])}]"
        )
        table4_rows.append(
            [
                labels[method],
                _number(row["AUTC_mean"]),
                _number(row["AUTC_sample_sd"]),
                "--" if comp is None else _number(comp["paired_mean"]),
                "--" if comp is None else _number(comp["paired_sd"]),
                ci,
                "--" if comp is None else _number(comp["effect_size_dz"], 4),
                _wilcoxon_text(comp),
            ]
        )
    table4_path = output / "table4_limo_ablation.tex"
    _write(
        table4_path,
        _tabular(
            "@{}lrrrrlrr@{}",
            ["Method", "Mean", "SD", r"Paired $\Delta$", r"SD$_\Delta$", "95\\% CI", r"$d_z$", "Exact $p$"],
            table4_rows,
        ),
    )

    recovery_rows = []
    for method in order:
        row = methods[method]
        recovery_rows.append(
            [
                labels[method],
                f"{_number(row['recovery_at_80_mean'], 1)} $\\pm$ {_number(row['recovery_at_80_sample_sd'], 1)}",
                f"{_number(row['recovery_at_160_mean'], 1)} $\\pm$ {_number(row['recovery_at_160_sample_sd'], 1)}",
                f"{_number(row['recovery_at_240_mean'], 1)} $\\pm$ {_number(row['recovery_at_240_sample_sd'], 1)}",
                f"{_number(row['recovery_at_320_mean'], 1)} $\\pm$ {_number(row['recovery_at_320_sample_sd'], 1)}",
            ]
        )
    recovery_path = output / "table4_limo_recovery.tex"
    _write(recovery_path, _tabular("@{}lrrrr@{}", ["Method", "@80", "@160", "@240", "@320"], recovery_rows))

    group_rows = [
        [
            _tex(row["display_label"]),
            row["group_count"],
            row["singleton_group_count"],
            _number(float(row["singleton_group_fraction"]) * 100, 1),
            row["maximum_group_size"],
            _number(row["formal_top_b_concentration_max"], 3),
            _number(row["minimum_margin_score"], 3),
            row["correction_rounds_total"],
            row["effective_replacements_total"],
            _number(row["AUTC_mean"]),
        ]
        for row in groups
    ]
    group_table_path = output / "table5_mn_group_key.tex"
    _write(
        group_table_path,
        _tabular(
            "@{}lrrrrrrrrr@{}",
            ["Group key", "Groups", "Singletons", "Singleton \\%", "Max size", r"Max $G_t$", r"Min $M_t$", "Corr.", "Repl.", "AUTC"],
            group_rows,
        ),
    )

    routing_rows = [
        [
            labels[method],
            _number(methods[method]["direct_rounds_mean"], 1),
            _number(methods[method]["correction_rounds_mean"], 1),
            _number(methods[method]["effective_replacements_mean"], 1),
            _number(methods[method]["mean_unique_groups_per_batch_mean"], 3),
            _number(methods[method]["repetition_rate_mean"], 3),
        ]
        for method in order
    ]
    routing_path = output / "table6_routing.tex"
    _write(routing_path, _tabular("@{}lrrrrr@{}", ["Method", "Direct", "Correction", "Replacements", "Unique groups/batch", "Repetition rate"], routing_rows))

    mc_table_path = output / "table_s_mc_dropout.tex"
    _write(
        mc_table_path,
        _tabular(
            "@{}rrrrrrr@{}",
            [r"$K$", r"Median $\rho$", "Top-$b$ overlap", "Gate-flip rate", r"Mean $|\Delta\mathrm{AUTC}|$", "Runtime (s)", "Runtime ratio"],
            [
                [
                    row["mc_passes"],
                    _number(row["median_uncertainty_spearman_vs_k30"], 4),
                    _number(row["median_top_b_overlap_vs_k30"], 4),
                    _number(row["gate_flip_rate_vs_k30"], 4),
                    _number(row["mean_absolute_AUTC_difference_vs_k30"], 5),
                    _number(row["mean_runtime_seconds"], 1),
                    _number(row["median_runtime_ratio_vs_k30"], 4),
                ]
                for row in mc_rows
            ],
        ),
    )

    dft_table_path = output / "table7_dft_provisional.tex"
    dft_verification_complete = _dft_verification_complete(dft_rows)
    dft_banner = (
        "% COMPLETED VERIFICATION: C120/C214 values are generated from the "
        "approved relaxations and dependent frozen statics.\n"
        if dft_verification_complete
        else "% PENDING VERIFICATION: C120/C214 values must be regenerated "
        "after the approved relaxations/statics.\n"
    )
    _write(
        dft_table_path,
        dft_banner
        + _tabular(
            "@{}llll@{}",
            ["Candidate", "Formula/ID", "Selected tested initialization", r"Current $E_f$ (eV atom$^{-1}$)"],
            [
                [
                    _tex(row["candidate_label"]),
                    _tex(row["candidate_id"]),
                    _tex(row["selected_magnetic_initialization"]),
                    _number(row["recomputed_formation_energy_eV_per_atom"], 9),
                ]
                for row in dft_rows
            ],
        ),
    )
    return [macro_path, table4_path, recovery_path, group_table_path, routing_path, mc_table_path, dft_table_path]


def _optional_table(
    source: Path,
    output: Path,
    columns: list[tuple[str, str]],
    *,
    longtable: bool = True,
) -> Path | None:
    if not source.exists():
        return None
    rows = _read_csv(source)
    table_rows = [[_tex(row.get(key, "")) for key, _ in columns] for row in rows]
    column_spec = "@{}" + "l" * len(columns) + "@{}"
    _write(output, _tabular(column_spec, [label for _, label in columns], table_rows, longtable=longtable))
    return output


def _build_supplementary_tables(bundle: Path, supplemental: Path, output: Path) -> list[Path]:
    generated: list[Path] = []
    checkpoint_source = supplemental / "tables" / "table_s_checkpoint_provenance.csv"
    if checkpoint_source.exists():
        checkpoint = _read_csv(checkpoint_source)[0]
        checkpoint_target = output / "table_s_checkpoint_provenance.tex"
        checkpoint_rows = [
            ["Checkpoint path", _parbox(r"0.72\linewidth", _path(checkpoint.get("checkpoint_path")))],
            ["SHA-256", _parbox(r"0.72\linewidth", _seqsplit(checkpoint.get("sha256")))],
            ["Normalizer mean", _tex(checkpoint.get("normalizer_mean"))],
            ["Normalizer scale", _tex(checkpoint.get("normalizer_scale"))],
            ["Exact CIF-hash overlaps", _tex(checkpoint.get("exact_cif_hash_overlap_count"))],
        ]
        _write(checkpoint_target, _tabular("@{}ll@{}", ["Field", "Recorded value"], checkpoint_rows))
        generated.append(checkpoint_target)

    manifest_source = bundle / "dft" / "dft_candidate_manifest.csv"
    if manifest_source.exists():
        manifest_rows = _read_csv(manifest_source)
        timeline_target = output / "table_s_dft_manifest.tex"
        status_target = output / "table_s_dft_manifest_status.tex"
        provenance_target = output / "table_s_dft_manifest_provenance.tex"
        _write(
            timeline_target,
            _tabular(
                "@{}lllllll@{}",
                ["Candidate", "Formula", "Cohort", "Freeze timestamp", "Known", "Gate", "Greedy"],
                [
                    [
                        _candidate_display(row.get("candidate_id", "")),
                        _tex(row.get("formula")),
                        _tex(row.get("pilot_or_new")),
                        _tex(row.get("freeze_timestamp")),
                        _tex(row.get("result_known_at_freeze")),
                        _tex(row.get("Gate_round")),
                        _tex(row.get("Greedy_round")),
                    ]
                    for row in manifest_rows
                ],
                longtable=True,
            ),
        )
        _write(
            status_target,
            _tabular(
                "@{}lllll@{}",
                ["Candidate", "Selection rule", "DFT status", "Failure reason", "Main text"],
                [
                    [
                        _candidate_display(row.get("candidate_id", "")),
                        _parbox(r"0.34\linewidth", _tex(row.get("selection_rule"))),
                        _parbox(r"0.15\linewidth", _tex(row.get("DFT_status"))),
                        _parbox(r"0.18\linewidth", _tex(row.get("failure_reason"))),
                        _tex(row.get("main_text_selected")),
                    ]
                    for row in manifest_rows
                ],
                longtable=True,
            ),
        )
        _write(
            provenance_target,
            _tabular(
                "@{}lll@{}",
                ["Candidate", "Timestamp source", "Artifact SHA-256"],
                [
                    [
                        _candidate_display(row.get("candidate_id", "")),
                        _parbox(r"0.50\linewidth", _tex(row.get("timestamp_source"))),
                        _parbox(r"0.26\linewidth", _seqsplit(row.get("sha256"))),
                    ]
                    for row in manifest_rows
                ],
                longtable=True,
            ),
        )
        generated.extend([timeline_target, status_target, provenance_target])

    reference_source = bundle / "dft" / "elemental_references.csv"
    if reference_source.exists():
        references = _read_csv(reference_source)
        reference_target = output / "table_s_elemental_references.tex"
        reference_provenance_target = output / "table_s_elemental_reference_provenance.tex"
        _write(
            reference_target,
            _tabular(
                "@{}llllllll@{}",
                ["Element", "Functional", "Structure", "Magnetic setup", r"$U_{\mathrm{eff}}$", "Mesh", "Energy/atom", "Converged"],
                [
                    [
                        _tex(row.get("element")),
                        _tex(row.get("functional")),
                        _parbox(r"0.13\linewidth", _tex(row.get("structure"))),
                        _parbox(r"0.22\linewidth", _tex(row.get("magnetic_setup"))),
                        _tex(row.get("Ueff_eV")),
                        _tex(row.get("kpoints_mesh")),
                        _tex(row.get("energy_per_atom_eV")),
                        _tex(row.get("electronic_converged")),
                    ]
                    for row in references
                ],
                longtable=True,
            ),
        )
        _write(
            reference_provenance_target,
            _tabular(
                "@{}llllll@{}",
                ["Element", "Functional", "PAW label", "Raw output", "OUTCAR SHA-256", "Limitation"],
                [
                    [
                        _tex(row.get("element")),
                        _tex(row.get("functional")),
                        _parbox(r"0.17\linewidth", _tex(row.get("paw_label"))),
                        _parbox(r"0.31\linewidth", _path(row.get("raw_output_path"))),
                        _parbox(r"0.18\linewidth", _seqsplit(row.get("raw_output_sha256"))),
                        _parbox(r"0.12\linewidth", _tex(row.get("reference_risk"))),
                    ]
                    for row in references
                ],
                longtable=True,
            ),
        )
        generated.extend([reference_target, reference_provenance_target])

    structure_source = bundle / "dft" / "structure_metrics.csv"
    if structure_source.exists():
        structures = _read_csv(structure_source)
        volume_target = output / "table_s_structure_metrics.tex"
        geometry_target = output / "table_s_structure_geometry_forces.tex"
        _write(
            volume_target,
            _tabular(
                "@{}lllllll@{}",
                ["Candidate", "Functional", "Initialization", r"Initial $V$", r"Final $V$", r"$\Delta V$ (\%)", "Space group"],
                [
                    [
                        _candidate_display(row.get("candidate_id", "")),
                        _tex(row.get("functional")),
                        _tex(row.get("magnetic_initialization")),
                        _tex(row.get("historical_selected_configuration_initial_volume_A3")),
                        _tex(row.get("historical_selected_configuration_final_volume_A3")),
                        _tex(row.get("historical_selected_configuration_relative_volume_change_percent")),
                        _tex(row.get("final_space_group")),
                    ]
                    for row in structures
                ],
                longtable=True,
            ),
        )
        _write(
            geometry_target,
            _tabular(
                "@{}lllllll@{}",
                ["Candidate", "Functional", "Initialization", "Min pair", "Min M--O", r"Relax $F_{\max}$", r"Static $F_{\max}$"],
                [
                    [
                        _candidate_display(row.get("candidate_id", "")),
                        _tex(row.get("functional")),
                        _tex(row.get("magnetic_initialization")),
                        _tex(row.get("minimum_interatomic_distance_A")),
                        _tex(row.get("minimum_M_O_distance_A")),
                        _tex(row.get("historical_selected_configuration_relaxation_Fmax_eV_A")),
                        _tex(row.get("Fmax_eV_A_static_diagnostic")),
                    ]
                    for row in structures
                ],
                longtable=True,
            ),
        )
        generated.extend([volume_target, geometry_target])

    verification_relaxation = bundle / "dft" / "verification_relaxation_metrics.csv"
    if verification_relaxation.exists():
        rows = _read_csv(verification_relaxation)
        lattice_target = output / "table_s_verification_lattice.tex"
        angles_target = output / "table_s_verification_angles.tex"
        geometry_target = output / "table_s_verification_geometry.tex"
        _write(
            lattice_target,
            _tabular(
                "@{}llrrrrrrrr@{}",
                [
                    "Candidate",
                    "Initialization",
                    r"$a_i$",
                    r"$b_i$",
                    r"$c_i$",
                    r"$a_f$",
                    r"$b_f$",
                    r"$c_f$",
                    r"$V_i$",
                    r"$V_f$",
                ],
                [
                    [
                        _tex(row.get("candidate_id")),
                        _tex(row.get("magnetic_initialization")),
                        _number(row.get("initial_a_A"), 4),
                        _number(row.get("initial_b_A"), 4),
                        _number(row.get("initial_c_A"), 4),
                        _number(row.get("final_a_A"), 4),
                        _number(row.get("final_b_A"), 4),
                        _number(row.get("final_c_A"), 4),
                        _number(row.get("initial_volume_A3"), 4),
                        _number(row.get("final_volume_A3"), 4),
                    ]
                    for row in rows
                ],
                longtable=True,
            ),
        )
        _write(
            angles_target,
            _tabular(
                "@{}llrrrrrr@{}",
                [
                    "Candidate",
                    "Initialization",
                    r"$\alpha_i$",
                    r"$\beta_i$",
                    r"$\gamma_i$",
                    r"$\alpha_f$",
                    r"$\beta_f$",
                    r"$\gamma_f$",
                ],
                [
                    [
                        _tex(row.get("candidate_id")),
                        _tex(row.get("magnetic_initialization")),
                        _number(row.get("initial_alpha_deg"), 4),
                        _number(row.get("initial_beta_deg"), 4),
                        _number(row.get("initial_gamma_deg"), 4),
                        _number(row.get("final_alpha_deg"), 4),
                        _number(row.get("final_beta_deg"), 4),
                        _number(row.get("final_gamma_deg"), 4),
                    ]
                    for row in rows
                ],
                longtable=True,
            ),
        )
        _write(
            geometry_target,
            _tabular(
                "@{}llrrrrll@{}",
                [
                    "Candidate",
                    "Initialization",
                    r"$\Delta V$ (\%)",
                    r"Max disp. (\AA)",
                    r"Min pair (\AA)",
                    r"Min M--O (\AA)",
                    r"$F_{\max}$",
                    r"Space group $i\rightarrow f$",
                ],
                [
                    [
                        _tex(row.get("candidate_id")),
                        _tex(row.get("magnetic_initialization")),
                        _number(row.get("relative_volume_change_percent"), 4),
                        _number(row.get("maximum_internal_displacement_A"), 5),
                        _number(row.get("minimum_interatomic_distance_A"), 4),
                        _number(row.get("minimum_M_O_distance_A"), 4),
                        _number(row.get("Fmax_eV_A"), 5),
                        _tex(
                            f"{row.get('initial_space_group', '')} -> "
                            f"{row.get('final_space_group', '')}"
                        ),
                    ]
                    for row in rows
                ],
                longtable=True,
            ),
        )
        generated.extend([lattice_target, angles_target, geometry_target])

    verification_static = bundle / "dft" / "verification_static_metrics.csv"
    if verification_static.exists():
        rows = _read_csv(verification_static)
        static_target = output / "table_s_verification_statics.tex"
        _write(
            static_target,
            _tabular(
                "@{}llrrrlll@{}",
                [
                    "Candidate",
                    "Initialization",
                    r"$E$ (eV)",
                    r"Diagnostic $F_{\max}$",
                    r"Moment ($\mu_B$)",
                    "Mesh",
                    "Space group",
                    "Converged",
                ],
                [
                    [
                        _tex(row.get("candidate_id")),
                        _tex(row.get("magnetic_initialization")),
                        _number(row.get("final_total_energy_eV"), 8),
                        _number(row.get("Fmax_eV_A_static_diagnostic"), 5),
                        _number(row.get("final_total_magnetic_moment"), 5),
                        _tex(row.get("kpoints_mesh")),
                        _tex(row.get("final_space_group")),
                        _tex(row.get("electronic_converged")),
                    ]
                    for row in rows
                ],
                longtable=True,
            ),
        )
        generated.append(static_target)

    selected_source = bundle / "dft" / "selected_candidate_comparison.csv"
    if selected_source.exists():
        rows = _read_csv(selected_source)
        selected_target = output / "table_s_verification_energy_shift.tex"
        _write(
            selected_target,
            _tabular(
                "@{}lllrrr@{}",
                [
                    "Candidate",
                    "Historical init.",
                    "Verification init.",
                    r"Historical $E_f$",
                    r"Verification $E_f$",
                    r"Shift",
                ],
                [
                    [
                        _tex(row.get("candidate_id")),
                        _tex(row.get("historical_selected_initialization")),
                        _tex(row.get("new_selected_initialization")),
                        _number(
                            row.get(
                                "historical_selected_formation_energy_eV_per_atom"
                            ),
                            9,
                        ),
                        _number(
                            row.get("new_selected_formation_energy_eV_per_atom"), 9
                        ),
                        _number(
                            row.get("selected_formation_energy_shift_eV_per_atom"),
                            9,
                        ),
                    ]
                    for row in rows
                ],
                longtable=True,
            ),
        )
        generated.append(selected_target)

    definitions = [
        (
            supplemental / "tables" / "table_s_parameter_calibration.csv",
            output / "table_s_parameter_calibration.tex",
            [("stage", "Stage"), ("cohort", "Cohort"), ("selection_rank", "Rank"), ("config_id", "Configuration"), ("M0", r"$M_0$"), ("G0", r"$G_0$"), ("alpha", r"$\alpha$"), ("beta", r"$\beta$"), ("gamma", r"$\gamma$")],
        ),
        (
            bundle / "gpu" / "per_seed_metrics.csv",
            output / "table_s_per_seed_metrics.tex",
            [("dataset", "Dataset"), ("method", "Method"), ("group_key", "Group key"), ("seed", "Seed"), ("K", r"$K$"), ("AUTC", "AUTC"), ("recovery_at_80", "@80"), ("recovery_at_160", "@160"), ("recovery_at_240", "@240"), ("recovery_at_320", "@320")],
        ),
        (
            bundle / "dft" / "magnetic_initializations.csv",
            output / "table_s_magnetic_initializations.tex",
            [("candidate_id", "Candidate"), ("magnetic_initialization", "Initialization"), ("final_total_energy_eV", "Total energy"), ("selected_lower_energy_among_two_tested", "Selected"), ("energy_difference_from_lower_eV", r"$\Delta E$") , ("scope_statement", "Scope")],
        ),
    ]
    for source, target, columns in definitions:
        result = _optional_table(source, target, columns)
        if result is not None:
            generated.append(result)
    return generated


def build_v34_tables(bundle: Path, supplemental: Path, output: Path) -> list[dict[str, str]]:
    bundle = Path(bundle)
    supplemental = Path(supplemental)
    output = Path(output)
    generated = _build_main_tables(bundle, output)
    generated.extend(_build_supplementary_tables(bundle, supplemental, output))
    source_paths = [
        bundle / "gpu" / "method_summary.csv",
        bundle / "gpu" / "paired_statistics.json",
        bundle / "gpu" / "mn_group_key_sensitivity_summary.csv",
        bundle / "gpu" / "mc_dropout_summary.csv",
        bundle / "dft" / "main_text_table7_comparison.csv",
        bundle / "dft" / "dft_candidate_manifest.csv",
        supplemental / "tables" / "table_s_parameter_calibration.csv",
        supplemental / "tables" / "table_s_checkpoint_provenance.csv",
    ]
    source_manifest = [
        {"source": path.as_posix(), "sha256": _sha256(path)}
        for path in source_paths
        if path.exists()
    ]
    manifest_path = output / "source_evidence_manifest.json"
    _write(manifest_path, json.dumps(source_manifest, indent=2, ensure_ascii=False))
    output_manifest = [
        {"path": path.as_posix(), "sha256": _sha256(path)}
        for path in sorted([*generated, manifest_path])
    ]
    _write(output / "generated_sha256.json", json.dumps(output_manifest, indent=2))
    return source_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--supplemental", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_v34_tables(args.bundle, args.supplemental, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
