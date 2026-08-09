"""Audit CGCNN checkpoint provenance and reproduce formal round-one predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scalar(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu().item()
    return float(value)


def extract_checkpoint_metadata(checkpoint_path: Path) -> dict[str, Any]:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = checkpoint.get("args", {})
    if hasattr(args, "__dict__"):
        args = vars(args)
    if not isinstance(args, dict):
        args = {}
    normalizer = checkpoint.get("normalizer", {})
    state_dict = checkpoint.get("state_dict", {})
    atom_features = int(args.get("atom_fea_len", 64))
    convolutions = int(args.get("n_conv", 3))
    hidden_features = int(args.get("h_fea_len", 128))
    hidden_layers = int(args.get("n_h", 1))
    return {
        "checkpoint_path": checkpoint_path.as_posix(),
        "sha256": sha256_file(checkpoint_path),
        "file_size_bytes": checkpoint_path.stat().st_size,
        "epoch": int(checkpoint.get("epoch", -1)),
        "best_mae_error": _scalar(checkpoint.get("best_mae_error", float("nan"))),
        "normalizer_mean": _scalar(normalizer["mean"]),
        "normalizer_scale": _scalar(normalizer["std"]),
        "training_target": str(args.get("task", "unavailable")),
        "training_data_options": json.dumps(args.get("data_options", [])),
        "train_ratio": float(args.get("train_ratio", float("nan"))),
        "val_ratio": float(args.get("val_ratio", float("nan"))),
        "test_ratio": float(args.get("test_ratio", float("nan"))),
        "epochs": int(args.get("epochs", -1)),
        "batch_size": int(args.get("batch_size", -1)),
        "learning_rate": float(args.get("lr", float("nan"))),
        "optimizer": str(args.get("optim", "unavailable")),
        "model_architecture": (
            f"CGCNN({atom_features},{convolutions},{hidden_features},{hidden_layers})"
        ),
        "atom_fea_len": atom_features,
        "n_conv": convolutions,
        "h_fea_len": hidden_features,
        "n_h": hidden_layers,
        "state_dict_layer_count": len(state_dict),
        "checkpoint_keys": ";".join(sorted(checkpoint.keys())),
    }


def exact_cif_hash_overlaps(training_dir: Path, pool_dir: Path) -> pd.DataFrame:
    training_by_hash: dict[str, list[str]] = defaultdict(list)
    for path in sorted(training_dir.glob("*.cif")):
        training_by_hash[sha256_file(path)].append(path.stem)
    rows: list[dict[str, str]] = []
    for pool_path in sorted(pool_dir.glob("*.cif")):
        digest = sha256_file(pool_path)
        for training_id in training_by_hash.get(digest, []):
            rows.append(
                {
                    "training_candidate_id": training_id,
                    "pool_candidate_id": pool_path.stem,
                    "cif_sha256": digest,
                }
            )
    return pd.DataFrame(
        rows,
        columns=["training_candidate_id", "pool_candidate_id", "cif_sha256"],
    )


def structure_matcher_overlaps(training_dir: Path, pool_dir: Path) -> pd.DataFrame:
    """Test same-reduced-formula pairs under disclosed StructureMatcher defaults."""

    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Structure

    matcher = StructureMatcher(
        ltol=0.2,
        stol=0.3,
        angle_tol=5,
        primitive_cell=True,
        scale=True,
        attempt_supercell=False,
        allow_subset=False,
    )
    training: dict[str, list[tuple[Path, Any]]] = defaultdict(list)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for path in sorted(training_dir.glob("*.cif")):
            structure = Structure.from_file(path)
            training[structure.composition.reduced_formula].append((path, structure))
        rows: list[dict[str, Any]] = []
        for pool_path in sorted(pool_dir.glob("*.cif")):
            pool_structure = Structure.from_file(pool_path)
            formula = pool_structure.composition.reduced_formula
            for training_path, training_structure in training.get(formula, []):
                if matcher.fit(training_structure, pool_structure):
                    rms = matcher.get_rms_dist(training_structure, pool_structure)
                    rows.append(
                        {
                            "training_candidate_id": training_path.stem,
                            "pool_candidate_id": pool_path.stem,
                            "reduced_formula": formula,
                            "rms_distance": float(rms[0]) if rms is not None else np.nan,
                            "max_distance": float(rms[1]) if rms is not None else np.nan,
                        }
                    )
    return pd.DataFrame(
        rows,
        columns=[
            "training_candidate_id",
            "pool_candidate_id",
            "reduced_formula",
            "rms_distance",
            "max_distance",
        ],
    )


def rank_interval_scores(ids: Iterable[str], scores: Iterable[float]) -> dict[str, int]:
    pairs = [(str(candidate_id), float(score)) for candidate_id, score in zip(ids, scores, strict=True)]
    if len({candidate_id for candidate_id, _ in pairs}) != len(pairs):
        raise ValueError("candidate IDs must be unique")
    if not all(np.isfinite(score) for _, score in pairs):
        raise ValueError("ranking scores must be finite")
    ordered = sorted(pairs, key=lambda pair: (-pair[1], pair[0]))
    return {candidate_id: rank for rank, (candidate_id, _) in enumerate(ordered, start=1)}


def archived_first_round_table(score_path: Path) -> pd.DataFrame:
    score = pd.read_csv(score_path)
    ranks = rank_interval_scores(score["id"], score["P_hit"])
    return pd.DataFrame(
        {
            "candidate_id": score["id"].astype(str),
            "mu_eV": pd.to_numeric(score["mu_eV"], errors="raise"),
            "sigma_eV": pd.to_numeric(score["sigma_eV"], errors="raise"),
            "interval_hit_score": pd.to_numeric(score["P_hit"], errors="raise"),
            "rank": score["id"].astype(str).map(ranks).astype(int),
        }
    )


def reproduce_first_round_predictions(
    *,
    formal_project: Path,
    checkpoint_path: Path,
    pool_dir: Path,
    experiment_seed: int,
    acquisition_round: int,
    model_refit_index: int,
    mc_passes: int,
    dropout_rate: float,
    target_low_ev: float,
    target_high_ev: float,
    sigma_floor_ev: float,
    device_name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    import torch
    from torch.utils.data import DataLoader

    project_text = str(formal_project.resolve())
    if project_text not in sys.path:
        sys.path.insert(0, project_text)
    from active_learning_etdg_tage import (  # type: ignore[import-not-found]
        CIFData,
        clean_id,
        collate_pool,
        load_model,
        penultimate_forward,
    )
    from mc_dropout_seed_policy import (  # type: ignore[import-not-found]
        SEED_POLICY_VERSION,
        mask_sequence_sha256,
        mc_mask_seeds,
    )
    from uncertainty_units import interval_hit_probability_ev  # type: ignore[import-not-found]

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA reproduction requested but CUDA is unavailable")
    device = torch.device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    location = _scalar(checkpoint["normalizer"]["mean"])
    scale = _scalar(checkpoint["normalizer"]["std"])
    model = load_model(str(checkpoint_path), device)
    dataset = CIFData(str(pool_dir))
    dataloader = DataLoader(
        dataset, batch_size=32, collate_fn=collate_pool, shuffle=False, num_workers=0
    )
    mask_seeds = mc_mask_seeds(
        mc_passes,
        experiment_seed=experiment_seed,
        acquisition_round=acquisition_round,
        model_refit_index=model_refit_index,
    )
    mask_hash = mask_sequence_sha256(
        mc_passes,
        experiment_seed=experiment_seed,
        acquisition_round=acquisition_round,
        model_refit_index=model_refit_index,
    )

    rows: list[dict[str, float | str]] = []
    keep_probability = 1.0 - float(dropout_rate)
    with torch.no_grad():
        for batch in dataloader:
            (atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx), _, batch_ids = batch
            ids = [clean_id(value) for value in batch_ids]
            atom_fea = atom_fea.to(device)
            nbr_fea = nbr_fea.to(device)
            nbr_fea_idx = nbr_fea_idx.to(device)
            crystal_atom_idx = [index.to(device) for index in crystal_atom_idx]
            embedding = penultimate_forward(
                model, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx
            ).detach()
            deterministic_normalized = model.fc_out(embedding).view(-1).detach().cpu().numpy()
            deterministic_ev = deterministic_normalized * scale + location
            predictions = []
            for mask_seed in mask_seeds:
                generator = torch.Generator(device=device)
                generator.manual_seed(mask_seed)
                random_values = torch.rand(
                    embedding.shape,
                    dtype=embedding.dtype,
                    device=device,
                    generator=generator,
                )
                mask = (random_values < keep_probability).to(embedding.dtype) / keep_probability
                predictions.append(model.fc_out(embedding * mask).view(-1).detach().cpu().numpy())
            matrix = np.vstack(predictions)
            sigma_normalized = np.std(matrix, axis=0, ddof=0)
            sigma_ev = np.abs(scale) * sigma_normalized
            interval_score = interval_hit_probability_ev(
                deterministic_ev,
                sigma_ev,
                target_low_ev,
                target_high_ev,
                sigma_floor_ev=sigma_floor_ev,
            )
            for candidate_id, mu_ev, sd_ev, p_hit in zip(
                ids, deterministic_ev, sigma_ev, interval_score, strict=True
            ):
                rows.append(
                    {
                        "candidate_id": candidate_id,
                        "mu_eV": float(mu_ev),
                        "sigma_eV": float(sd_ev),
                        "interval_hit_score": float(p_hit),
                    }
                )
    frame = pd.DataFrame(rows)
    ranks = rank_interval_scores(frame["candidate_id"], frame["interval_hit_score"])
    frame["rank"] = frame["candidate_id"].map(ranks).astype(int)
    metadata = {
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "mc_mask_sequence_sha256": mask_hash,
        "mc_seed_policy_version": SEED_POLICY_VERSION,
        "candidate_count": len(frame),
        "normalizer_mean": location,
        "normalizer_scale": scale,
    }
    return frame, metadata


def compare_first_round_predictions(
    reproduced: pd.DataFrame, archived: pd.DataFrame, *, atol: float
) -> pd.DataFrame:
    required = {"candidate_id", "mu_eV", "sigma_eV", "interval_hit_score", "rank"}
    if not required.issubset(reproduced.columns) or not required.issubset(archived.columns):
        raise ValueError("prediction tables are missing required columns")
    reproduced_ids = set(reproduced["candidate_id"].astype(str))
    archived_ids = set(archived["candidate_id"].astype(str))
    if reproduced_ids != archived_ids:
        raise ValueError("candidate ID sets differ between reproduced and archived tables")
    left = reproduced.copy()
    right = archived.copy()
    left["candidate_id"] = left["candidate_id"].astype(str)
    right["candidate_id"] = right["candidate_id"].astype(str)
    compared = left.merge(right, on="candidate_id", suffixes=("", "_archived"), validate="one_to_one")
    compared = compared.sort_values("candidate_id").reset_index(drop=True)
    delta_columns = []
    for column in ("mu_eV", "sigma_eV", "interval_hit_score"):
        delta = f"{column}_delta"
        compared[delta] = compared[column] - compared[f"{column}_archived"]
        delta_columns.append(delta)
    compared["within_tolerance"] = compared[delta_columns].abs().le(float(atol)).all(axis=1)
    compared["rank_matches"] = compared["rank"].astype(int).eq(
        compared["rank_archived"].astype(int)
    )
    return compared


def _write_without_overwriting_different_content(frame: pd.DataFrame, output: Path) -> None:
    rendered = frame.to_csv(index=False, lineterminator="\n")
    if output.exists():
        if output.read_text(encoding="utf-8") != rendered:
            raise FileExistsError(f"refusing to overwrite non-identical evidence: {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8", newline="")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    formal_project = root / "artifacts/gpu_server/completed_formal_results/63729a5a4bea44b3/attempt_1/payload/project"
    formal_run = root / "artifacts/gpu_server/completed_formal_results/63729a5a4bea44b3/attempt_1/payload/results/final/li_m_o_ablation/energy_gated_da_tpp/seed_15/attempt_1"
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-project", type=Path, default=formal_project)
    parser.add_argument("--checkpoint", type=Path, default=formal_project / "checkpoint_formation_clean.pth.tar")
    parser.add_argument("--training-dir", type=Path, default=Path(r"D:\CGCNN\mp_formation_clean"))
    parser.add_argument(
        "--pool-dir",
        type=Path,
        default=formal_project / "EARLY_LiM2O4_ALIGNN_HARD640_m2p18_m2p02_FLAT_20260617",
    )
    parser.add_argument(
        "--archived-score",
        type=Path,
        default=formal_run / "energy_gated_da_tpp_scores_egdatpp_psfix_v1_iter_1.csv",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument(
        "--prediction-output",
        type=Path,
        default=root / "results/reproducibility/first_round_predictions.csv",
    )
    parser.add_argument(
        "--provenance-output",
        type=Path,
        default=root / "supplementary/tables/table_s_checkpoint_provenance.csv",
    )
    args = parser.parse_args()

    checkpoint_metadata = extract_checkpoint_metadata(args.checkpoint)
    exact_overlap = exact_cif_hash_overlaps(args.training_dir, args.pool_dir)
    near_overlap = structure_matcher_overlaps(args.training_dir, args.pool_dir)
    reproduced, run_metadata = reproduce_first_round_predictions(
        formal_project=args.formal_project,
        checkpoint_path=args.checkpoint,
        pool_dir=args.pool_dir,
        experiment_seed=15,
        acquisition_round=1,
        model_refit_index=0,
        mc_passes=30,
        dropout_rate=0.3,
        target_low_ev=-2.18,
        target_high_ev=-2.02,
        sigma_floor_ev=0.05,
        device_name=args.device,
    )
    archived = archived_first_round_table(args.archived_score)
    compared = compare_first_round_predictions(reproduced, archived, atol=args.atol)
    _write_without_overwriting_different_content(compared, args.prediction_output)

    provenance = {
        **checkpoint_metadata,
        "checkpoint_classification": "project_trained_not_official_pretrained",
        "source_code_repository": "https://github.com/txie-93/cgcnn.git",
        "checkpoint_generation_script": "D:/CGCNN/launch_clean_training.py",
        "training_data_generation_script": "D:/CGCNN/prepare_mp_formation_clean.py",
        "training_id_prop_path": "D:/CGCNN/mp_formation_clean/id_prop.csv",
        "training_id_prop_sha256": sha256_file(args.training_dir / "id_prop.csv"),
        "training_row_count": sum(1 for _ in (args.training_dir / "id_prop.csv").open(encoding="utf-8")),
        "split_partition_rule": "first 80% train; final 20% validation; 0% test",
        "split_random_seed": "unavailable_not_explicitly_set",
        "exact_cif_hash_overlap_count": len(exact_overlap),
        "structure_matcher_overlap_count": len(near_overlap),
        "structure_matcher_parameters": "ltol=0.2;stol=0.3;angle_tol=5;primitive_cell=true;scale=true;attempt_supercell=false;allow_subset=false",
        "loaded_layers": "all checkpoint state_dict layers (strict load_state_dict)",
        "first_round_frozen": True,
        "later_round_training": "all model parameters fine-tuned during scheduled refits",
        "first_round_reproduction_device": run_metadata["device"],
        "first_round_reproduction_torch": run_metadata["torch_version"],
        "first_round_candidate_count": len(compared),
        "first_round_all_within_tolerance": bool(compared["within_tolerance"].all()),
        "first_round_all_ranks_match": bool(compared["rank_matches"].all()),
        "first_round_max_abs_mu_delta_eV": float(compared["mu_eV_delta"].abs().max()),
        "first_round_max_abs_sigma_delta_eV": float(compared["sigma_eV_delta"].abs().max()),
        "first_round_max_abs_probability_delta": float(compared["interval_hit_score_delta"].abs().max()),
        "mc_mask_sequence_sha256": run_metadata["mc_mask_sequence_sha256"],
        "archived_score_path": args.archived_score.as_posix(),
        "archived_score_sha256": sha256_file(args.archived_score),
    }
    _write_without_overwriting_different_content(pd.DataFrame([provenance]), args.provenance_output)
    print(json.dumps(provenance, indent=2, default=str))


if __name__ == "__main__":
    main()
