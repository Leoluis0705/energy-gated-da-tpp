from pathlib import Path
import shutil

import pandas as pd
import pytest
from PIL import Image

from analysis.build_v34_figures import build_v34_figure, build_v34_figures


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_v34_figure_data_use_frozen_formal_cohorts(tmp_path: Path) -> None:
    package = build_v34_figures(REPO_ROOT, tmp_path)

    figure3 = pd.read_csv(package.source_data["Figure3"])
    formal = figure3.loc[figure3["record_type"] == "trajectory"]
    assert set(formal["seed"].dropna().astype(int)) == set(range(15, 25))
    assert set(formal["method"]) == {
        "always_da_tpp",
        "energy_gated_da_tpp",
        "group_only_gate",
        "interval_hit_greedy",
        "margin_only_gate",
    }

    full = formal.loc[formal["method"] == "energy_gated_da_tpp"].sort_values(
        ["seed", "oracle_evaluations"]
    )
    group = formal.loc[formal["method"] == "group_only_gate"].sort_values(
        ["seed", "oracle_evaluations"]
    )
    assert full["cumulative_target_count"].tolist() == group[
        "cumulative_target_count"
    ].tolist()

    figure5 = pd.read_csv(package.source_data["Figure5"])
    summary = figure5.loc[figure5["record_type"] == "group_summary"]
    assert set(summary["group_key"]) == {
        "element_system_current",
        "coelement_block_multiset",
        "coelement_iupac_group_set",
    }
    assert (summary["correction_rounds_total"].astype(float) == 0).all()
    assert (summary["effective_replacements_total"].astype(float) == 0).all()


def test_v34_figures_include_statistics_pending_state_and_submission_formats(
    tmp_path: Path,
) -> None:
    package = build_v34_figures(REPO_ROOT, tmp_path)

    figure4 = pd.read_csv(package.source_data["Figure4"])
    full = figure4.loc[
        (figure4["record_type"] == "paired_statistic")
        & (figure4["method"] == "energy_gated_da_tpp")
    ].iloc[0]
    assert float(full["paired_mean"]) == pytest.approx(0.010160256410256419)
    assert float(full["bootstrap_low"]) == pytest.approx(0.0028205128205128216)
    assert float(full["bootstrap_high"]) == pytest.approx(0.018044871794871787)
    assert float(full["exact_wilcoxon_p"]) == pytest.approx(0.0546875)
    assert float(full["dz"]) == pytest.approx(0.7825080231134547)

    figure6 = pd.read_csv(package.source_data["Figure6"])
    status = figure6.loc[figure6["record_type"] == "main_candidate_metric"].set_index(
        "candidate_label"
    )["verification_status"]
    assert status["C044"] == "archived assessment"
    assert status["C120"] == "verification relaxation pending"
    assert status["C214"] == "verification relaxation pending"

    for figure_number in range(1, 7):
        outputs = package.figures[f"Figure{figure_number}"]
        assert {path.suffix for path in outputs} == {".pdf", ".svg", ".png"}
        assert all(path.exists() and path.stat().st_size > 0 for path in outputs)
        png = next(path for path in outputs if path.suffix == ".png")
        with Image.open(png) as image:
            dpi = image.info.get("dpi", (0, 0))
            assert min(dpi) >= 599

    figure1_audit = package.qa["Figure1"]
    assert figure1_audit["content_layout_changed"] is False
    assert figure1_audit["native_vector"] is False
    assert figure1_audit["native_width_px"] == 1536
    assert figure1_audit["native_height_px"] == 1024


def test_each_figure_has_an_independent_rebuild_entry_point(tmp_path: Path) -> None:
    package = build_v34_figure(2, REPO_ROOT, tmp_path)
    assert set(package.figures) == {"Figure2"}
    assert set(package.source_data) == {"Figure2"}
    for figure_number in range(1, 7):
        script = (
            REPO_ROOT
            / "manuscript"
            / "v34_working_source"
            / "Scripts"
            / f"plot_figure{figure_number}.py"
        )
        assert script.exists()


def test_figure6_uses_completed_verification_metrics_from_explicit_bundle(
    tmp_path: Path,
) -> None:
    historical = (
        REPO_ROOT
        / "results"
        / "post_submission_analysis"
        / "egdatpp_psfix_v1_20260719T031102Z"
    )
    bundle = tmp_path / "verification_bundle"
    shutil.copytree(historical, bundle)

    main_path = bundle / "dft" / "main_text_table7_comparison.csv"
    main = pd.read_csv(main_path)
    completed = main["candidate_label"].isin(["C120", "C214"])
    main.loc[completed, "verification_status"] = (
        "completed_frozen_protocol_relaxation_and_static"
    )
    main.to_csv(main_path, index=False)

    structure_path = bundle / "dft" / "structure_metrics.csv"
    structure = pd.read_csv(structure_path)
    verified = structure["candidate_id"].astype(str).str.contains(
        r"job_(?:120|214)_", regex=True
    )
    structure.loc[verified, "verification_relative_volume_change_percent"] = 0.125
    structure.loc[verified, "verification_relaxation_Fmax_eV_A"] = 0.031
    structure.to_csv(structure_path, index=False)

    package = build_v34_figure(
        6,
        REPO_ROOT,
        tmp_path / "output",
        bundle_override=bundle,
    )
    source = pd.read_csv(package.source_data["Figure6"])
    metrics = source.loc[source["record_type"] == "main_candidate_metric"].set_index(
        "candidate_label"
    )
    assert metrics.loc["C120", "verification_status"] == (
        "completed frozen-protocol verification"
    )
    assert metrics.loc["C214", "verification_status"] == (
        "completed frozen-protocol verification"
    )
    assert float(metrics.loc["C120", "relaxation_volume_change_percent"]) == pytest.approx(
        0.125
    )
    assert float(metrics.loc["C214", "relaxation_Fmax_eV_A"]) == pytest.approx(0.031)
    assert package.qa["Figure6"]["pending_candidates"] == []
