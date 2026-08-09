from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from analysis.audit_common import (
    bootstrap_mean_ci,
    candidate_sequence_hash,
    first_attainment_positions,
    left_continuous_autc,
    recovery_at,
    sha256_file,
    write_bytes_protected,
)


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "evidence.bin"
    path.write_bytes(b"audit-evidence\n")
    assert sha256_file(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_left_continuous_autc_uses_previous_checkpoint_recovery() -> None:
    value = left_continuous_autc(
        query_counts=np.array([2, 4]),
        recoveries=np.array([1, 3]),
        total_targets=4,
        budget=4,
    )
    assert value == 0.125


def test_left_continuous_autc_extends_last_recovery_to_budget() -> None:
    value = left_continuous_autc(
        query_counts=np.array([2, 4]),
        recoveries=np.array([1, 3]),
        total_targets=4,
        budget=6,
    )
    assert value == 8 / 24


def test_recovery_at_uses_last_completed_checkpoint() -> None:
    queries = np.array([16, 32, 48])
    recoveries = np.array([3, 7, 10])
    assert recovery_at(queries, recoveries, 8) == 0
    assert recovery_at(queries, recoveries, 32) == 7
    assert recovery_at(queries, recoveries, 40) == 7


def test_candidate_sequence_hash_is_order_sensitive_and_newline_delimited() -> None:
    expected = hashlib.sha256(b"a\nb\nc\n").hexdigest()
    assert candidate_sequence_hash(["a", "b", "c"]) == expected
    assert candidate_sequence_hash(["a", "c", "b"]) != expected


def test_first_attainment_positions_uses_candidate_level_order() -> None:
    assert first_attainment_positions([0, 1, 0, 1, 1]) == {1: 2, 2: 4, 3: 5}


def test_bootstrap_mean_ci_is_deterministic_and_percentile_based() -> None:
    first = bootstrap_mean_ci([1.0, 2.0, 4.0], samples=10_000, seed=123)
    second = bootstrap_mean_ci([1.0, 2.0, 4.0], samples=10_000, seed=123)
    assert first == second
    assert first == (1.0, 4.0)


def test_write_bytes_protected_refuses_silent_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "result.csv"
    assert write_bytes_protected(path, b"a,b\n1,2\n", check_existing=False) == "created"
    with np.testing.assert_raises(FileExistsError):
        write_bytes_protected(path, b"a,b\n1,2\n", check_existing=False)
    assert write_bytes_protected(path, b"a,b\n1,2\n", check_existing=True) == "verified_identical"
    with np.testing.assert_raises(RuntimeError):
        write_bytes_protected(path, b"a,b\n3,4\n", check_existing=True)


def test_write_bytes_protected_can_resume_only_missing_outputs(tmp_path: Path) -> None:
    existing = tmp_path / "existing.csv"
    missing = tmp_path / "missing.csv"
    existing.write_bytes(b"same\n")
    assert (
        write_bytes_protected(
            existing,
            b"same\n",
            check_existing=True,
            create_if_missing_during_check=True,
        )
        == "verified_identical"
    )
    assert (
        write_bytes_protected(
            missing,
            b"new\n",
            check_existing=True,
            create_if_missing_during_check=True,
        )
        == "created_missing_during_verified_resume"
    )
