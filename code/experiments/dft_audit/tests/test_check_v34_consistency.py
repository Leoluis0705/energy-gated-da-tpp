from pathlib import Path

from analysis.check_v34_consistency import (
    _dft_verification_boundary,
    audit_v34_package,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_v34_package_is_numerically_and_semantically_consistent() -> None:
    result = audit_v34_package(REPO_ROOT)
    assert result["passed"] is True
    assert result["checks"]["formal_seed_cohort"] == "PASS"
    assert result["checks"]["full_group_sequence_identity"] == "PASS"
    assert result["checks"]["paired_statistics"] == "PASS"
    assert result["checks"]["mn_group_key_direct_route"] == "PASS"
    assert result["checks"]["dft_verification_boundary"] == "PASS"
    assert result["checks"]["figure_formats_and_dpi"] == "PASS"
    assert result["checks"]["latex_logs"] == "PASS"
    assert result["v33_sha256"] == "070fe58f550723865f315922abd222c8f9f460cf8e9ef4f5c8fbb4af65f18cc0"


def test_consistency_report_uses_ascii_status_separator(tmp_path: Path) -> None:
    report = tmp_path / "consistency.md"
    audit_v34_package(REPO_ROOT, report_path=report)

    text = report.read_text(encoding="utf-8")
    assert "- **PASS - formal_seed_cohort:**" in text


def test_dft_verification_boundary_accepts_pending_or_completed_evidence() -> None:
    pending_status = {
        "C044": "archived assessment",
        "C120": "verification relaxation pending",
        "C214": "verification relaxation pending",
    }
    pending_body = (
        "C120/C214 verification results are pending; the DFT stage does not identify "
        "a DFT-confirmed exact-target material."
    )
    pending_si = "alpha-Mn remains not approved and pending."
    assert _dft_verification_boundary(pending_status, pending_body, pending_si) == (
        True,
        "pending",
    )

    completed_status = {
        "C044": "archived assessment",
        "C120": "completed frozen-protocol verification",
        "C214": "completed frozen-protocol verification",
    }
    completed_body = (
        "C120/C214 verification relaxations and dependent frozen statics completed; "
        "the DFT stage does not identify a DFT-confirmed exact-target material."
    )
    completed_si = "C120/C214 verification outputs are complete. alpha-Mn remains not approved."
    assert _dft_verification_boundary(
        completed_status, completed_body, completed_si
    ) == (True, "completed")
