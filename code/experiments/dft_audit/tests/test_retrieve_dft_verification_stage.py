from pathlib import PurePosixPath

import pytest

from analysis.retrieve_dft_verification_stage import (
    is_transfer_allowed,
    safe_local_target,
    terminal_watcher_state,
)


@pytest.mark.parametrize(
    "name",
    [
        "POTCAR",
        "POTCAR.Li_sv",
        "WAVECAR",
        "CHGCAR",
        "CHG",
        "AECCAR0",
        "AECCAR1",
        "AECCAR2",
    ],
)
def test_licensed_or_large_vasp_payloads_are_not_transferable(name):
    assert is_transfer_allowed(PurePosixPath("job") / name) is False


def test_required_text_evidence_and_final_cifs_are_transferable():
    for name in (
        "initial.cif",
        "POSCAR",
        "CONTCAR",
        "INCAR",
        "KPOINTS",
        "OUTCAR",
        "OSZICAR",
        "vasprun.xml",
        "run.log",
        "metrics.csv",
        "review.json",
    ):
        assert is_transfer_allowed(PurePosixPath("job") / name) is True


def test_local_path_mapping_rejects_escape(tmp_path):
    assert safe_local_target(tmp_path, "job/OUTCAR") == tmp_path / "job" / "OUTCAR"
    with pytest.raises(ValueError, match="unsafe remote relative path"):
        safe_local_target(tmp_path, "../POTCAR")
    with pytest.raises(ValueError, match="unsafe remote relative path"):
        safe_local_target(tmp_path, "/absolute/OUTCAR")


@pytest.mark.parametrize(
    "status,authorized",
    [
        ("POSTPROCESS_DONE", True),
        ("POSTPROCESS_DONE_PAPER_UPDATE_PAUSED", False),
        ("BLOCKED_BY_UPSTREAM_GATE", False),
    ],
)
def test_only_terminal_watcher_states_are_accepted(status, authorized):
    assert terminal_watcher_state({"status": status}) == (status, authorized)


def test_running_watcher_state_is_rejected():
    with pytest.raises(ValueError, match="not terminal"):
        terminal_watcher_state({"status": "WAITING_FOR_GATED_STATIC_COMPLETION"})
