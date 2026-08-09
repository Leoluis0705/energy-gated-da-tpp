from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from analysis.run_mc_dropout_microbenchmark import (
    loaded_formal_support_module,
    mc_mask_seeds,
    summarize_mc_sensitivity,
)
from analysis.run_gate_ablation_smoke import load_formal_gate_module


def test_mc_mask_seeds_are_nested_prefixes() -> None:
    seeds3 = mc_mask_seeds(3)
    seeds10 = mc_mask_seeds(10)
    seeds30 = mc_mask_seeds(30)
    assert seeds3 == [1_000_003, 1_000_004, 1_000_005]
    assert seeds10[:3] == seeds3
    assert seeds30[:10] == seeds10


def test_summary_reports_differences_overlap_and_gate_flip() -> None:
    ids = ["a", "b", "c", "d"]
    states = {
        3: pd.DataFrame({"id": ids, "mean_ev": [0.0, 1.0, 2.0, 3.0], "sd_ev": [0.1, 0.2, 0.3, 0.4]}),
        10: pd.DataFrame({"id": ids, "mean_ev": [0.0, 1.1, 2.0, 2.9], "sd_ev": [0.1, 0.25, 0.35, 0.4]}),
        30: pd.DataFrame({"id": ids, "mean_ev": [0.1, 1.0, 1.9, 3.0], "sd_ev": [0.4, 0.3, 0.2, 0.1]}),
    }
    selections = {
        3: {"top_b_ids": ["a", "b"], "full_ids": ["a", "b"], "route": "direct", "margin": 1.2, "concentration": 0.5},
        10: {"top_b_ids": ["a", "c"], "full_ids": ["a", "c"], "route": "direct", "margin": 1.1, "concentration": 0.5},
        30: {"top_b_ids": ["c", "d"], "full_ids": ["c", "d"], "route": "correction", "margin": 0.8, "concentration": 0.5},
    }
    result = summarize_mc_sensitivity(
        states,
        selections,
        runtimes={3: 1.0, 10: 2.0, 30: 3.0},
        peak_memory_mib={3: 10.0, 10: 11.0, 30: 12.0},
        embedding_runtime_seconds=4.0,
        embedding_peak_memory_mib=9.0,
    )
    assert result["mc_passes"].tolist() == [3, 10, 30]
    baseline = result[result["mc_passes"] == 3].iloc[0]
    assert baseline["mean_abs_predictive_mean_difference_ev_vs_k3"] == 0.0
    assert baseline["top_b_overlap_fraction_vs_k3"] == 1.0
    assert not bool(baseline["gate_flip_vs_k3"])
    k10 = result[result["mc_passes"] == 10].iloc[0]
    assert k10["top_b_overlap_fraction_vs_k3"] == 0.5
    k30 = result[result["mc_passes"] == 30].iloc[0]
    assert k30["uncertainty_rank_spearman_vs_k3"] == -1.0
    assert bool(k30["gate_flip_vs_k3"])
    assert k30["total_runtime_seconds_with_shared_embedding"] == 7.0


def test_script_supports_direct_cli_invocation() -> None:
    archive = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable, str(archive / "analysis/run_mc_dropout_microbenchmark.py"), "--help"],
        cwd=archive,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--development-seed" in completed.stdout


def test_formal_support_loader_exposes_model_and_dataset_interfaces() -> None:
    archive = Path(__file__).resolve().parents[2]
    load_formal_gate_module(
        archive / "active_learning_energy_gate_ablation.py",
        archive / "experiments/reproducibility/staging/paired_confirmation_server_20260712",
    )
    support = loaded_formal_support_module()
    for name in ("load_model", "CIFData", "collate_pool", "penultimate_forward"):
        assert hasattr(support, name)
