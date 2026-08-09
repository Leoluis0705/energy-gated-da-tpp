#!/usr/bin/env python3
"""Audit numerical, semantic, and file-format consistency of the v34 package."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


EXPECTED_V33_SHA256 = "070fe58f550723865f315922abd222c8f9f460cf8e9ef4f5c8fbb4af65f18cc0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _macro_values(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    return dict(re.findall(r"\\newcommand\{\\([^}]+)\}\{([^}]*)\}", text))


def _dft_verification_boundary(
    statuses: dict[str, str], body_tex: str, si_tex: str
) -> tuple[bool, str]:
    common = (
        statuses.get("C044") == "archived assessment"
        and "alpha-Mn" in si_tex
        and "does not identify a DFT-confirmed exact-target material" in body_tex
    )
    pending = (
        common
        and statuses.get("C120") == "verification relaxation pending"
        and statuses.get("C214") == "verification relaxation pending"
        and "C120/C214 verification results are pending" in body_tex
        and "pending" in si_tex.lower()
    )
    completed = (
        common
        and statuses.get("C120") == "completed frozen-protocol verification"
        and statuses.get("C214") == "completed frozen-protocol verification"
        and "C120/C214 verification relaxations and dependent frozen statics completed"
        in body_tex
        and "C120/C214 verification outputs are complete" in si_tex
        and "C120/C214 verification results are pending" not in body_tex
    )
    if pending:
        return True, "pending"
    if completed:
        return True, "completed"
    return False, "inconsistent"


def audit_v34_package(
    repo_root: Path,
    report_path: Path | None = None,
    bundle_override: Path | None = None,
) -> dict[str, object]:
    repo = Path(repo_root).resolve()
    work = repo / "manuscript" / "v34_working_source"
    source_dir = work / "SourceData"
    figure_dir = work / "Figures"
    bundle = (
        Path(bundle_override).resolve()
        if bundle_override is not None
        else repo
        / "results"
        / "post_submission_analysis"
        / "egdatpp_psfix_v1_dftverify_v2_20260719T114742Z"
    )
    checks: dict[str, str] = {}
    details: list[str] = []

    def record(name: str, condition: bool, detail: str) -> None:
        checks[name] = "PASS" if condition else "FAIL"
        details.append(f"- **{checks[name]} - {name}:** {detail}")

    figure3 = pd.read_csv(source_dir / "Figure3_source.csv")
    trajectories = figure3.loc[figure3["record_type"] == "trajectory"].copy()
    seeds = set(trajectories["seed"].dropna().astype(int))
    record(
        "formal_seed_cohort",
        seeds == set(range(15, 25)),
        f"Figure 3 formal trajectories contain seeds {sorted(seeds)} and no legacy formal cohort.",
    )
    per_seed = figure3.loc[figure3["record_type"] == "per_seed_metric"].copy()
    full_hash = per_seed.loc[per_seed["method"] == "energy_gated_da_tpp"].set_index("seed")["candidate_sequence_sha256"]
    group_hash = per_seed.loc[per_seed["method"] == "group_only_gate"].set_index("seed")["candidate_sequence_sha256"]
    identity = full_hash.index.equals(group_hash.index) and (full_hash == group_hash).all()
    record(
        "full_group_sequence_identity",
        bool(identity),
        "Full Gate and Group-only candidate-sequence SHA-256 values match seed by seed for all ten confirmatory runs.",
    )
    greedy_autc = per_seed.loc[per_seed["method"] == "interval_hit_greedy"].set_index("seed")["AUTC"].astype(float)
    margin_autc = per_seed.loc[per_seed["method"] == "margin_only_gate"].set_index("seed")["AUTC"].astype(float)
    record(
        "margin_greedy_autc_identity",
        bool(np.allclose(greedy_autc.sort_index(), margin_autc.sort_index(), atol=0, rtol=0)),
        "Margin-only and Greedy AUTC values are exactly equal for seeds 15--24.",
    )

    figure4 = pd.read_csv(source_dir / "Figure4_source.csv")
    stats = figure4.loc[figure4["record_type"] == "paired_statistic"].set_index("method")
    expected = {
        "paired_mean": 0.010160256410256419,
        "bootstrap_low": 0.0028205128205128216,
        "bootstrap_high": 0.018044871794871787,
        "exact_wilcoxon_p": 0.0546875,
        "dz": 0.7825080231134547,
    }
    full_stats = stats.loc["energy_gated_da_tpp"]
    statistic_match = all(np.isclose(float(full_stats[key]), value) for key, value in expected.items())
    macros = _macro_values(work / "Tables" / "generated" / "formal_metrics_macros.tex")
    macro_match = (
        np.isclose(float(macros["FullPairedMean"]), expected["paired_mean"], atol=5e-7)
        and np.isclose(float(macros["FullBootstrapLow"]), expected["bootstrap_low"], atol=5e-7)
        and np.isclose(float(macros["FullBootstrapHigh"]), expected["bootstrap_high"], atol=5e-7)
        and np.isclose(float(macros["FullWilcoxonP"]), expected["exact_wilcoxon_p"])
        and np.isclose(float(macros["FullDz"]), expected["dz"], atol=5e-5)
    )
    record(
        "paired_statistics",
        bool(statistic_match and macro_match),
        "Figure 4 source data and LaTeX macros agree on paired delta, 95% bootstrap CI, exact Wilcoxon p, and dz.",
    )

    figure5 = pd.read_csv(source_dir / "Figure5_source.csv")
    groups = figure5.loc[figure5["record_type"] == "group_summary"].copy()
    expected_keys = {"element_system_current", "coelement_block_multiset", "coelement_iupac_group_set"}
    group_direct = (
        set(groups["group_key"]) == expected_keys
        and (groups["correction_rounds_total"].astype(float) == 0).all()
        and (groups["effective_replacements_total"].astype(float) == 0).all()
        and (groups["minimum_margin_score"].astype(float) > 0.75).all()
    )
    record(
        "mn_group_key_direct_route",
        bool(group_direct),
        "All three preregistered Mn group keys have minimum M_t above 0.75, zero correction rounds, and zero replacements.",
    )

    main_tex = (work / "manuscript_v34.tex").read_text(encoding="utf-8")
    body_tex = (work / "v34_body.tex").read_text(encoding="utf-8")
    si_tex = (work / "supplementary.tex").read_text(encoding="utf-8")
    abstract = main_tex.split("\\abstract{", 1)[1].split("\\keyword", 1)[0].lower()
    conclusion = body_tex.split("\\section{Conclusions}", 1)[1].lower()
    language_ok = "statistically significant" not in abstract and "statistically significant" not in conclusion
    record(
        "inference_language",
        language_ok and "p = \\FullWilcoxonP" in main_tex,
        "Abstract and conclusion avoid a statistically-significant claim and the abstract reports the exact p-value relative to 0.05.",
    )

    figure6 = pd.read_csv(source_dir / "Figure6_source.csv")
    main_metrics = figure6.loc[figure6["record_type"] == "main_candidate_metric"].set_index("candidate_label")
    statuses = main_metrics["verification_status"].to_dict()
    dft_boundary, dft_mode = _dft_verification_boundary(statuses, body_tex, si_tex)
    table7 = pd.read_csv(bundle / "dft" / "main_text_table7_comparison.csv").set_index("candidate_label")
    dft_numeric = all(
        np.isclose(
            float(main_metrics.loc[label, "recomputed_formation_energy_eV_per_atom"]),
            float(table7.loc[label, "recomputed_formation_energy_eV_per_atom"]),
        )
        for label in ("C044", "C120", "C214")
    )
    record(
        "dft_verification_boundary",
        bool(dft_boundary and dft_numeric),
        f"DFT evidence mode is {dft_mode}; C044 remains an archived assessment, "
        "the three plotted energies match the selected evidence table, and no "
        "exact-target confirmation is claimed.",
    )

    format_ok = True
    format_detail: list[str] = []
    for number in range(1, 7):
        for suffix in (".pdf", ".svg", ".png"):
            path = figure_dir / f"Figure{number}_v34{suffix}"
            format_ok &= path.exists() and path.stat().st_size > 0
        png = figure_dir / f"Figure{number}_v34.png"
        with Image.open(png) as image:
            dpi = image.info.get("dpi", (0, 0))
            format_ok &= min(dpi) >= 599
            format_detail.append(f"F{number} {image.width}x{image.height}px @ {min(dpi):.1f} dpi")
    record(
        "figure_formats_and_dpi",
        bool(format_ok),
        "PDF, SVG and PNG exist for Figures 1--6; " + "; ".join(format_detail) + ".",
    )

    figure1 = pd.read_csv(source_dir / "Figure1_source.csv").iloc[0]
    figure1_ok = (
        str(figure1["native_vector"]).lower() == "false"
        and int(figure1["native_width_px"]) == 1536
        and int(figure1["native_height_px"]) == 1024
        and str(figure1["content_layout_changed"]).lower() == "false"
    )
    record(
        "figure1_audit",
        bool(figure1_ok),
        "Figure 1 content/layout are unchanged; native 1536x1024 raster and non-native-vector status are disclosed.",
    )

    logs_ok = True
    log_details = []
    for name in ("manuscript_v34.log", "supplementary.log"):
        log = (work / name).read_text(encoding="utf-8", errors="replace")
        bad = [token for token in ("undefined citations", "undefined references", "Overfull \\hbox", "! LaTeX Error") if token.lower() in log.lower()]
        logs_ok &= not bad
        log_details.append(f"{name}: {'clean' if not bad else ', '.join(bad)}")
    pdfs_ok = (work / "manuscript_v34.pdf").exists() and (work / "supplementary.pdf").exists()
    record("latex_logs", bool(logs_ok and pdfs_ok), "; ".join(log_details) + "; both PDFs exist.")

    v33 = Path.home() / "Desktop" / "Energy_Gated_DA_TPP_CMC_organized_v33.pdf"
    v33_sha = _sha256(v33)
    record(
        "v33_preservation",
        v33_sha == EXPECTED_V33_SHA256,
        f"Protected v33 PDF SHA-256 remains {v33_sha}.",
    )

    passed = all(value == "PASS" for value in checks.values())
    result: dict[str, object] = {
        "passed": passed,
        "checks": checks,
        "v33_sha256": v33_sha,
        "details": details,
    }
    if report_path is not None:
        report = Path(report_path)
        report.parent.mkdir(parents=True, exist_ok=True)
        title = "PASS" if passed else "FAIL"
        report.write_text(
            "# v34 Numerical Consistency Report\n\n"
            f"Overall result: **{title}**\n\n"
            "This audit compares frozen evidence tables, figure source CSVs, generated LaTeX macros, manuscript language, pending-calculation boundaries, submission figure formats, LaTeX logs, and the protected v33 hash.\n\n"
            + "\n".join(details)
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--bundle", type=Path)
    args = parser.parse_args()
    result = audit_v34_package(args.repo_root, args.report, args.bundle)
    print("PASS" if result["passed"] else "FAIL")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
