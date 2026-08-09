"""End-to-end three-system low-data multi-fidelity experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from analysis.three_system.data import (
    MODEL_FEATURE_COLUMNS,
    build_historical_binary_labels,
    build_three_system_pool,
    model_feature_frame,
)
from analysis.three_system.models import (
    bootstrap_binary_predictions,
    evaluate_binary_models_nested_loo,
    evaluate_energy_calibrators_loo,
    fit_energy_calibrator_predict,
)
from analysis.three_system.replay import (
    ALL_METHODS,
    freeze_prospective_union,
    run_paired_replay,
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def largest_remainder_quotas(
    counts: dict[str, int],
    total: int,
) -> dict[str, int]:
    """Allocate an integer sample proportionally with deterministic ties."""
    denominator = sum(counts.values())
    if denominator <= 0 or total <= 0:
        raise ValueError("counts and total must be positive")
    exact = {
        key: total * value / denominator for key, value in counts.items()
    }
    quotas = {key: int(np.floor(value)) for key, value in exact.items()}
    remaining = total - sum(quotas.values())
    order = sorted(
        counts,
        key=lambda key: (-(exact[key] - quotas[key]), key),
    )
    for key in order[:remaining]:
        quotas[key] += 1
    return quotas


def build_selector_equivalence_audit(
    selections: pd.DataFrame,
    *,
    left_method: str = "group_gated_da_tpp",
    right_method: str = "full_gate",
) -> pd.DataFrame:
    """Compare semantically identical selectors within every paired seed."""
    required = {
        "method",
        "seed",
        "query",
        "candidate_id",
        "proxy_model_seed",
    }
    missing = required - set(selections.columns)
    if missing:
        raise ValueError(f"selection audit is missing columns: {sorted(missing)}")

    rows = []
    for seed in sorted(selections["seed"].unique()):
        left = selections.loc[
            (selections["seed"] == seed)
            & (selections["method"] == left_method),
            ["query", "candidate_id", "proxy_model_seed"],
        ].sort_values("query", kind="mergesort")
        right = selections.loc[
            (selections["seed"] == seed)
            & (selections["method"] == right_method),
            ["query", "candidate_id", "proxy_model_seed"],
        ].sort_values("query", kind="mergesort")
        if left.empty or right.empty:
            raise ValueError(
                f"seed {seed} is missing {left_method} or {right_method}"
            )

        compared = left.merge(
            right,
            on="query",
            how="outer",
            suffixes=("_left", "_right"),
            indicator=True,
            validate="one_to_one",
        ).sort_values("query", kind="mergesort")
        candidate_equal = (
            compared["_merge"].eq("both")
            & compared["candidate_id_left"].eq(compared["candidate_id_right"])
        )
        model_seed_equal = (
            compared["_merge"].eq("both")
            & compared["proxy_model_seed_left"].eq(
                compared["proxy_model_seed_right"]
            )
        )
        difference = ~(candidate_equal & model_seed_equal)
        first_difference = (
            int(compared.loc[difference, "query"].iloc[0])
            if difference.any()
            else np.nan
        )
        rows.append(
            {
                "seed": int(seed),
                "left_method": left_method,
                "right_method": right_method,
                "same_candidate_sequence": bool(candidate_equal.all()),
                "same_proxy_model_seed_sequence": bool(model_seed_equal.all()),
                "first_difference_query": first_difference,
            }
        )
    return pd.DataFrame(rows)


def require_selector_equivalence(audit: pd.DataFrame) -> None:
    """Stop result publication when equivalent selectors diverge."""
    equivalent = (
        audit["same_candidate_sequence"].astype(bool)
        & audit["same_proxy_model_seed_sequence"].astype(bool)
    )
    failed = audit.loc[~equivalent]
    if failed.empty:
        return
    details = ", ".join(
        f"seed {int(row.seed)} at query {int(row.first_difference_query)}"
        for row in failed.itertuples(index=False)
    )
    raise RuntimeError(f"paired selector equivalence failed: {details}")


def build_energy_calibration_table(
    pool: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    quantitative = labels.loc[
        labels["dft_evaluable"].eq(1)
        & labels["dft_formation_energy_eV_atom"].notna(),
        ["candidate_id", "m_element", "dft_formation_energy_eV_atom"],
    ].copy()
    sources = pool.loc[
        :,
        [
            "candidate_id",
            "alignn_formation_energy_eV_atom",
            "chgnet_final_energy_eV_atom",
            "mace_final_energy_eV_atom",
        ],
    ]
    result = quantitative.merge(
        sources,
        on="candidate_id",
        validate="one_to_one",
    ).rename(
        columns={
            "dft_formation_energy_eV_atom": "dft_energy",
            "alignn_formation_energy_eV_atom": "alignn",
            "chgnet_final_energy_eV_atom": "chgnet",
            "mace_final_energy_eV_atom": "mace",
        }
    )
    if len(result) != 10:
        raise ValueError(f"expected ten quantitative DFT points, found {len(result)}")
    return result.sort_values("candidate_id", kind="mergesort").reset_index(drop=True)


def _verify_protocol_sources(root: Path, protocol: dict) -> None:
    for entry in protocol["source_inputs"].values():
        path = root / entry["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256_file(path)
        if observed != entry["sha256"]:
            raise ValueError(f"source SHA-256 mismatch for {path}: {observed}")


def _composition_probability(
    labels: pd.DataFrame,
    elements: pd.Series,
) -> np.ndarray:
    rows = []
    for element in elements.astype(str):
        local = labels.loc[
            labels["m_element"].astype(str) == element,
            "dft_evaluable",
        ].to_numpy(dtype=int)
        rows.append(float((local.sum() + 1.0) / (len(local) + 2.0)))
    return np.asarray(rows)


def _candidate_energy_input(pool: pd.DataFrame) -> pd.DataFrame:
    return pool.loc[
        :,
        [
            "candidate_id",
            "m_element",
            "alignn_formation_energy_eV_atom",
            "chgnet_final_energy_eV_atom",
            "mace_final_energy_eV_atom",
        ],
    ].rename(
        columns={
            "alignn_formation_energy_eV_atom": "alignn",
            "chgnet_final_energy_eV_atom": "chgnet",
            "mace_final_energy_eV_atom": "mace",
        }
    )


def _git_commit(root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _environment_payload() -> dict[str, object]:
    packages = {}
    for name in (
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "PyYAML",
        "joblib",
    ):
        packages[name] = importlib.metadata.version(name)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }


def summarize_replay_results(replay_results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate paired replay endpoints without dropping the joint target."""
    return (
        replay_results.groupby(["method", "checkpoint"], as_index=False)
        .agg(
            mean_estimated_DFT_evaluable_count=(
                "estimated_DFT_evaluable_count",
                "mean",
            ),
            sd_estimated_DFT_evaluable_count=(
                "estimated_DFT_evaluable_count",
                "std",
            ),
            mean_simulated_DFT_evaluable_count=(
                "simulated_DFT_evaluable_count",
                "mean",
            ),
            sd_simulated_DFT_evaluable_count=(
                "simulated_DFT_evaluable_count",
                "std",
            ),
            mean_estimated_interval_hit_count=(
                "estimated_interval_hit_count",
                "mean",
            ),
            sd_estimated_interval_hit_count=(
                "estimated_interval_hit_count",
                "std",
            ),
            mean_simulated_interval_hit_count=(
                "simulated_interval_hit_count",
                "mean",
            ),
            sd_simulated_interval_hit_count=(
                "simulated_interval_hit_count",
                "std",
            ),
            mean_unique_structure_clusters=("unique_structure_clusters", "mean"),
            mean_unique_compositions=("unique_compositions", "mean"),
        )
        .sort_values(["checkpoint", "method"], kind="mergesort")
        .reset_index(drop=True)
    )


def _write_go_no_go(
    output: Path,
    *,
    pool: pd.DataFrame,
    labels: pd.DataFrame,
    binary_summary: pd.DataFrame,
    energy_summary: pd.DataFrame,
    replay_summary: pd.DataFrame,
    frozen: pd.DataFrame,
    dft_submission_status: str,
) -> None:
    best_binary = binary_summary.iloc[0]
    best_energy = energy_summary.iloc[0]
    early = replay_summary.loc[replay_summary["checkpoint"] == 32].sort_values(
        "mean_simulated_interval_hit_count",
        ascending=False,
        kind="mergesort",
    )
    early_lines = "\n".join(
        f"| {row.method} | {row.mean_simulated_interval_hit_count:.3f} | "
        f"{row.mean_estimated_interval_hit_count:.3f} | "
        f"{row.mean_estimated_DFT_evaluable_count:.3f} | "
        f"{row.mean_unique_structure_clusters:.3f} |"
        for row in early.itertuples(index=False)
    )
    composition = pool["m_element"].value_counts()
    frozen_composition = frozen["m_element"].value_counts()
    text = "\n".join(
        [
            "# Three-System Low-Data Multi-Fidelity GO/NO-GO",
            "",
            "## Decision",
            "",
            "- **GO_FOR_RETROSPECTIVE_ML_ASSISTED_REPLAY**",
            f"- **{dft_submission_status}**",
            "- Manuscript remains unchanged.",
            "",
            "## Evidence boundary",
            "",
            f"- Frozen pool: {len(pool)} candidates "
            f"(Cr={composition.get('Cr', 0)}, Mn={composition.get('Mn', 0)}, "
            f"Mg={composition.get('Mg', 0)}).",
            f"- Historical DFT labels: {len(labels)} "
            f"({int(labels['dft_evaluable'].sum())} strict positives and "
            f"{int((1-labels['dft_evaluable']).sum())} strict negatives).",
            "- Prospective pool: 255 candidates.",
            "- High-fidelity interval remains frozen at [-2.3, -1.5] eV/atom "
            "and is part of the joint target.",
            "- The primary replay endpoint is the count jointly pseudo-labeled "
            "as DFT-evaluable and inside that interval. It is an ML-assisted "
            "simulation endpoint, not an observed DFT count.",
            "",
            "## Model validation",
            "",
            f"- Selected evaluability model: `{best_binary.model_name}`.",
            f"- Nested-LOO ROC-AUC: {best_binary.loo_roc_auc:.3f}; balanced "
            f"accuracy: {best_binary.loo_balanced_accuracy:.3f}; Brier: "
            f"{best_binary.loo_brier_score:.3f}; log loss: "
            f"{best_binary.loo_log_loss:.3f}.",
            f"- Selected energy calibrator: `{best_energy.model_id}`.",
            f"- LOO energy MAE: {best_energy.loo_mae_eV_atom:.4f} eV/atom; "
            f"RMSE: {best_energy.loo_rmse_eV_atom:.4f} eV/atom.",
            "- The binary dataset has only 20 observations and the energy "
            "dataset only 10; uncertainty and composition-stratified results "
            "must accompany any use.",
            "",
            "## ML-labeled target-qualified yield at query 32",
            "",
            "| Method | Mean ML-labeled qualified | Mean expected qualified | "
            "Mean expected evaluable | Mean structure clusters |",
            "|---|---:|---:|---:|---:|",
            early_lines,
            "",
            "These are paired pseudo-oracle replay estimates, not observed DFT outcomes.",
            "",
            "## Frozen prospective DFT union",
            "",
            f"- Candidates: {len(frozen)} (Cr={frozen_composition.get('Cr', 0)}, "
            f"Mn={frozen_composition.get('Mn', 0)}, Mg={frozen_composition.get('Mg', 0)}).",
            f"- Submission status: `{dft_submission_status}`.",
            "- Shared candidates are to be computed once and credited at each "
            "policy's own first-selection point.",
            "",
            "## Claims not yet supported",
            "",
            "- No new prospective candidate is counted as truly DFT-evaluable.",
            "- ML pseudo-labels are not DFT measurements.",
            "- HF-AUC versus actual core-hours cannot be calculated until new "
            "DFT jobs finish.",
            "- The frozen formation-energy interval is not claimed to be an "
            "engineering optimum or a stability criterion.",
            "",
        ]
    )
    (output / "THREE_SYSTEM_GO_NO_GO.md").write_text(
        text, encoding="utf-8", newline="\n"
    )


def run_pipeline(
    root: Path,
    output: Path,
    *,
    replay_bootstrap_draws: int | None = None,
) -> dict[str, object]:
    protocol_path = root / "THREE_SYSTEM_PROTOCOL_FREEZE.yaml"
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    _verify_protocol_sources(root, protocol)
    output.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    log_lines = [f"{started.isoformat()} pipeline_start"]

    pool = build_three_system_pool(
        root / protocol["source_inputs"]["candidate_pool_master"]["path"],
        root / protocol["source_inputs"]["mlip_full_pool_results"]["path"],
        root / protocol["source_inputs"]["historical_dft_manifest"]["path"],
    )
    labels = build_historical_binary_labels(
        root / protocol["source_inputs"]["historical_dft_manifest"]["path"],
        root / protocol["source_inputs"]["recomputed_formation_energies"]["path"],
    )
    pool.to_csv(output / "three_system_pool.csv", index=False)
    labels.to_csv(output / "historical_dft_binary_labels.csv", index=False)
    log_lines.append(
        f"{datetime.now(timezone.utc).isoformat()} pool_and_labels_complete"
    )

    history = pool.merge(
        labels[["candidate_id", "dft_evaluable"]],
        on="candidate_id",
        how="inner",
        validate="one_to_one",
    )
    history_x = model_feature_frame(history)
    history_y = history["dft_evaluable"].to_numpy(dtype=int)
    binary_predictions, binary_summary = evaluate_binary_models_nested_loo(
        history_x,
        history_y,
        random_seed=20260727,
    )
    binary_predictions["candidate_id"] = binary_predictions["row_index"].map(
        history["candidate_id"]
    )
    binary_summary.to_csv(output / "dft_evaluability_model_cv.csv", index=False)
    binary_predictions.to_csv(
        output / "dft_evaluability_model_oof_predictions.csv", index=False
    )
    selected_binary_model = str(binary_summary.iloc[0]["model_name"])
    log_lines.append(
        f"{datetime.now(timezone.utc).isoformat()} binary_cv_complete "
        f"selected={selected_binary_model}"
    )

    calibration = build_energy_calibration_table(pool, labels)
    energy_predictions, energy_summary = evaluate_energy_calibrators_loo(
        calibration,
        interval=tuple(protocol["high_fidelity_interval_eV_atom"]),
        bootstrap_draws=10000,
        random_seed=20260727,
    )
    energy_summary.to_csv(
        output / "formation_energy_calibration_cv.csv", index=False
    )
    energy_predictions.to_csv(
        output / "formation_energy_calibration_oof_predictions.csv", index=False
    )
    selected_energy_model = str(energy_summary.iloc[0]["model_id"])
    selected_alignn_model = str(
        energy_summary.loc[energy_summary["source"] == "ALIGNN"].iloc[0]["model_id"]
    )
    selected_mlip_model = str(
        energy_summary.loc[energy_summary["source"] != "ALIGNN"].iloc[0]["model_id"]
    )
    log_lines.append(
        f"{datetime.now(timezone.utc).isoformat()} energy_cv_complete "
        f"selected={selected_energy_model}"
    )

    prospective = pool.loc[pool["prospective_eligible"]].copy().reset_index(
        drop=True
    )
    prospective_x = model_feature_frame(prospective)
    candidate_draws = int(
        protocol["retrospective_replay"]["candidate_score_bootstrap_draws"]
    )
    binary_scores = bootstrap_binary_predictions(
        history_x,
        history_y,
        prospective_x,
        model_name=selected_binary_model,
        random_seed=20260727,
        draws=candidate_draws,
    )
    energy_input = _candidate_energy_input(prospective)
    interval = tuple(protocol["high_fidelity_interval_eV_atom"])
    primary_energy = fit_energy_calibrator_predict(
        calibration,
        energy_input,
        model_id=selected_energy_model,
        interval=interval,
    )
    alignn_energy = fit_energy_calibrator_predict(
        calibration,
        energy_input,
        model_id=selected_alignn_model,
        interval=interval,
    ).rename(columns={"p_interval_hit": "p_interval_alignn"})
    mlip_energy = fit_energy_calibrator_predict(
        calibration,
        energy_input,
        model_id=selected_mlip_model,
        interval=interval,
    ).rename(columns={"p_interval_hit": "p_interval_mlip"})

    scores = prospective.loc[
        :,
        [
            "candidate_id",
            "m_element",
            "formula",
            "cif_sha256",
            "structure_matcher_cluster",
            "structure_matcher_cluster_size",
            "atom_count",
            "space_group_number",
            *MODEL_FEATURE_COLUMNS,
        ],
    ].copy()
    scores = scores.loc[:, ~scores.columns.duplicated()]
    scores["p_dft_evaluable"] = binary_scores.mean
    scores["evaluability_uncertainty"] = binary_scores.standard_deviation
    scores["composition_only_probability"] = _composition_probability(
        labels, scores["m_element"]
    )
    scores = scores.merge(
        primary_energy[
            [
                "candidate_id",
                "predicted_dft_energy_mean",
                "predicted_dft_energy_std",
                "p_interval_hit",
                "energy_calibration_model_id",
            ]
        ],
        on="candidate_id",
        validate="one_to_one",
    )
    scores = scores.merge(
        alignn_energy[["candidate_id", "p_interval_alignn"]],
        on="candidate_id",
        validate="one_to_one",
    )
    scores = scores.merge(
        mlip_energy[["candidate_id", "p_interval_mlip"]],
        on="candidate_id",
        validate="one_to_one",
    )
    scores["dft_evaluability_model"] = selected_binary_model
    scores["evidence_tier"] = "ML_estimate_only"
    scores.to_csv(output / "three_system_candidate_scores.csv", index=False)
    log_lines.append(
        f"{datetime.now(timezone.utc).isoformat()} candidate_scoring_complete"
    )

    replay_config = protocol["retrospective_replay"]
    per_round_draws = (
        int(replay_bootstrap_draws)
        if replay_bootstrap_draws is not None
        else int(replay_config["per_round_proxy_bootstrap_draws"])
    )
    replay_results, replay_selections = run_paired_replay(
        scores,
        history_x,
        history_y,
        model_feature_columns=tuple(MODEL_FEATURE_COLUMNS),
        paired_seeds=tuple(replay_config["paired_seeds"]),
        methods=ALL_METHODS,
        batch_size=int(replay_config["batch_size"]),
        query_budget=int(replay_config["query_budget"]),
        checkpoints=tuple(replay_config["checkpoints"]),
        classifier_model_name=selected_binary_model,
        bootstrap_draws=per_round_draws,
    )
    replay_results.to_csv(
        output / "paired_baseline_replay_results.csv", index=False
    )
    replay_selections.to_csv(
        output / "paired_baseline_replay_selections.csv", index=False
    )
    selector_audit = build_selector_equivalence_audit(replay_selections)
    require_selector_equivalence(selector_audit)
    selector_audit.to_csv(
        output / "paired_selector_equivalence_audit.csv",
        index=False,
    )
    replay_summary = summarize_replay_results(replay_results)
    replay_summary.to_csv(
        output / "paired_baseline_replay_summary.csv", index=False
    )
    log_lines.append(
        f"{datetime.now(timezone.utc).isoformat()} paired_replay_complete"
    )

    early = replay_selections.loc[
        replay_selections["query"]
        <= int(protocol["prospective_dft"]["selection_horizon_queries"])
    ].copy()
    prospective_counts = prospective["m_element"].value_counts().to_dict()
    quotas = largest_remainder_quotas(
        prospective_counts,
        int(protocol["prospective_dft"]["maximum_unique_candidates"]),
    )
    frozen = freeze_prospective_union(
        early,
        maximum_candidates=int(
            protocol["prospective_dft"]["maximum_unique_candidates"]
        ),
        required_elements=tuple(protocol["prospective_dft"]["required_elements"]),
        composition_quotas=quotas,
    )
    prior_cr_campaign = {
        "job_079_Cr_fe_-0.854_n4_generated_crystals_cif__gen_1",
        "job_092_Cr_fe_-1.075_n4_generated_crystals_cif__gen_3",
        "job_126_Cr_fe_-0.901_n4_generated_crystals_cif__gen_0",
        "job_196_Cr_fe_-0.819_n4_generated_crystals_cif__gen_1",
        "job_234_Cr_fe_-1.123_n4_generated_crystals_cif__gen_3",
    }
    frozen["preexisting_cr_campaign_candidate"] = frozen["candidate_id"].isin(
        prior_cr_campaign
    )
    frozen["submission_status"] = "FROZEN_NOT_SUBMITTED"
    frozen["observed_DFT_result_available"] = False
    frozen["evidence_tier"] = "prospective_candidate_freeze_only"
    frozen.to_csv(output / "prospective_dft_union_12.csv", index=False)
    log_lines.append(
        f"{datetime.now(timezone.utc).isoformat()} prospective_union_frozen"
    )

    dft_submission_status = "NEW_DFT_NOT_SUBMITTED_PENDING_SERVER_STATUS"
    _write_go_no_go(
        output,
        pool=pool,
        labels=labels,
        binary_summary=binary_summary,
        energy_summary=energy_summary,
        replay_summary=replay_summary,
        frozen=frozen,
        dft_submission_status=dft_submission_status,
    )
    shutil.copy2(protocol_path, output / "THREE_SYSTEM_PROTOCOL_FREEZE.yaml")
    environment = _environment_payload()
    (output / "environment_lock.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    requirements = "\n".join(
        f"{name}=={version}"
        for name, version in environment["packages"].items()
    )
    (output / "requirements-lock.txt").write_text(
        requirements + "\n", encoding="utf-8", newline="\n"
    )
    metadata = {
        "started_utc": started.isoformat(),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(root),
        "selected_binary_model": selected_binary_model,
        "selected_energy_model": selected_energy_model,
        "candidate_score_bootstrap_draws": candidate_draws,
        "per_round_proxy_bootstrap_draws": per_round_draws,
        "new_dft_jobs_submitted": 0,
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    log_lines.append(f"{datetime.now(timezone.utc).isoformat()} pipeline_complete")
    (output / "three_system_pipeline.log").write_text(
        "\n".join(log_lines) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    manifest_rows = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.csv":
            manifest_rows.append(
                {
                    "relative_path": path.relative_to(output).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    pd.DataFrame(manifest_rows).to_csv(output / "SHA256SUMS.csv", index=False)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--output-dir",
        default="results/three_system_low_data_v1",
    )
    parser.add_argument("--replay-bootstrap-draws", type=int)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output = (root / args.output_dir).resolve()
    metadata = run_pipeline(
        root,
        output,
        replay_bootstrap_draws=args.replay_bootstrap_draws,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
