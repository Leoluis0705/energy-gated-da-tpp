# Energy-Gated DA-TPP

This repository contains the data, code, frozen protocols, and analysis assets
for:

> **Energy-Gated DA-TPP: Batch Active Learning for Early Target Recovery in Generated Crystal Pools**

The implemented task is early recovery of candidates whose formation energies
fall inside a prescribed target interval. The active-learning workflow begins
from the frozen candidate pool. Reference labels remain hidden until a batch has
been selected. The DFT-evaluability model, MLIP calculations, and DFT audit are
downstream analyses and never affect acquisition.

## Experiment workflow

The main Li--M--O experiment follows this sequence:

1. Load the frozen 640-candidate CIF pool, its candidate-ID table, the initial
   CGCNN checkpoint, and the hidden reference-label table.
2. Predict the remaining pool with CGCNN and use seeded MC dropout to obtain a
   predictive mean and spread for every candidate.
3. Convert each predictive distribution into an interval-hit probability
   `P_hit` for the target window `[-2.18, -2.02] eV atom^-1`.
4. Form the direct top-`b` front with `b = 16`, then calculate the normalized
   front margin `M_t` and the largest element-system concentration `G_t`.
5. Keep the direct batch when `M_t >= M0` and `G_t <= G0`. Otherwise rebuild
   the batch sequentially with

   ```text
   score_i = P_hit_i + alpha * uncertainty_i
             - beta * max_similarity_i - gamma * group_reuse_i
   ```

6. Reveal reference labels only for the selected batch, append those candidates
   to the labelled set, and refit CGCNN for 10 epochs.
7. Repeat prediction, acquisition, label reveal, and refitting until the query
   budget is reached.
8. Aggregate paired Gate and Predicted-Target Greedy trajectories at query
   budgets 80, 160, 240, and 320; compute normalized AUTC; then rebuild the
   manuscript figures and tables from the retained source data.

The frozen main-evaluation protocol uses seeds 15--24, 30 MC-dropout passes,
`M0 = 1.0`, `G0 = 0.5`, `alpha = 0.1`, `beta = 0.2`, and `gamma = 0.05`.
Gate and Greedy share the same seed-specific training and inference schedule.

## Program implementation

The active-learning implementation is under
`code/experiments/active_learning/`:

- `EARLY_LiM2O4_ALIGNN_HARD640_m2p18_m2p02_FLAT_20260617/` is the frozen
  640-CIF candidate pool, and
  `EARLY_LiM2O4_ALIGNN_HARD640_m2p18_m2p02_20260617/oracle.csv` is the
  ID-aligned reference-label table revealed only after selection.
- `checkpoint_formation_clean.pth.tar` is the common initial CGCNN checkpoint.
- `main.py` and `cgcnn/` implement CGCNN training, checkpoint loading, and graph
  data handling.
- `experiments/reproducibility/paired_predict_no_shuffle.py` performs
  deterministic candidate-ID-aligned inference.
- `mc_dropout_protocol.py` defines the seeded MC-dropout schedule and converts
  normalized predictive spread to `eV atom^-1` before evaluating interval-hit
  probabilities.
- `active_learning_energy_gate_ablation.py` implements the direct front, the
  `M_t`/`G_t` gate, diversity-aware batch construction, Greedy and ablation
  routes, and per-round score/trace artifacts.
- `experiments/reproducibility/run_paired_dataset_job.py` is the trajectory
  runner. It copies the pool into an isolated run directory, predicts, selects,
  reveals queried labels, retrains, checkpoints every round, and writes the
  final metrics.
- `experiments/reproducibility/formal_protocol.py` validates method, seed,
  cohort, parameter, and group-key boundaries before a run starts.
- `configs/frozen_protocols/egdatpp_psfix_v1/limo_gamma005_heldout.yaml` is the
  frozen Gate--Greedy main-evaluation protocol.
- `manuscript/Scripts/build_v60_gamma005_holdout.py` audits and aggregates the
  20 main-evaluation trajectories.
- `manuscript/Scripts/rebuild_v69_holdout_figure.py` rebuilds the held-out
  comparison figure from the retained per-seed source table.

The runner records `run_config.json`, `environment.json`, `command.txt`,
`al_history.csv`, per-round predictions and scores, route traces, checkpoint
hashes, `summary.csv`, `run_metrics.csv`, and `status.json`. Candidate IDs are
explicitly reindexed after prediction so that model output order cannot change
the acquisition order.

## Repository map

- `code/experiments/active_learning/` -- CGCNN, acquisition methods, frozen
  pools, protocols, checkpoint, analysis code, and tests.
- `code/experiments/hidden_evaluability/` -- acquisition-blind workflow
  evaluability model and leave-one-out analysis.
- `code/experiments/dft_audit/` -- first-principles input preparation,
  convergence checks, and formation-energy post-processing.
- `data/archived_single_run/` -- the six archived single-run query histories.
- `data/source_tables/` -- bounded tables used by the paper analyses.
- `manuscript/SourceData/` -- figure- and table-level source data.
- `manuscript/Scripts/` -- manuscript analysis and plotting programs.
- `scripts/validate_repository.py` -- repository integrity, focused scientific
  tests, and figure rebuilding.
- `provenance/` -- SHA-256 manifests and audit records.

## Environment

Python 3.11 is the tested interpreter.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Complete CGCNN trajectories require a CUDA-enabled PyTorch installation and a
GPU. Protocol tests, source-data checks, statistical aggregation, DFT
post-processing, and figure rebuilding are CPU-compatible.

## Validate the repository

From the repository root:

```bash
python scripts/validate_repository.py --tests
python scripts/validate_repository.py --figures
```

The first command checks required files, information-boundary paths, and the
focused active-learning, hidden-evaluability, and DFT test suites. The second
rebuilds the manuscript figures from retained source data.

## Run one main-evaluation trajectory

From `code/experiments/active_learning`:

```bash
python experiments/reproducibility/run_paired_dataset_job.py \
  --project-root . \
  --dataset limo \
  --method energy_gated_da_tpp \
  --seed 15 \
  --run-dir results/main_evaluation/energy_gated_da_tpp/seed_15/attempt_1 \
  --protocol-config configs/frozen_protocols/egdatpp_psfix_v1/limo_gamma005_heldout.yaml
```

Use `--method predicted_target_greedy` with the same seed and protocol for the
paired direct-probability baseline. A completed run has `"status": "DONE"` in
`status.json`; a failed run retains its logs and reports `"status": "FAILED"`.

## Run all paired main-evaluation trajectories

The main comparison contains two methods for each seed from 15 through 24. On
Bash, run from `code/experiments/active_learning`:

```bash
for seed in {15..24}; do
  for method in energy_gated_da_tpp predicted_target_greedy; do
    python experiments/reproducibility/run_paired_dataset_job.py \
      --project-root . \
      --dataset limo \
      --method "$method" \
      --seed "$seed" \
      --run-dir "results/main_evaluation/$method/seed_$seed/attempt_1" \
      --protocol-config configs/frozen_protocols/egdatpp_psfix_v1/limo_gamma005_heldout.yaml
  done
done
```

The equivalent PowerShell loop is:

```powershell
15..24 | ForEach-Object {
  $seed = $_
  "energy_gated_da_tpp", "predicted_target_greedy" | ForEach-Object {
    $method = $_
    python experiments/reproducibility/run_paired_dataset_job.py `
      --project-root . `
      --dataset limo `
      --method $method `
      --seed $seed `
      --run-dir "results/main_evaluation/$method/seed_$seed/attempt_1" `
      --protocol-config configs/frozen_protocols/egdatpp_psfix_v1/limo_gamma005_heldout.yaml
  }
}
```

The loops are sequential and intentionally keep each run in its own directory.
Independent jobs may be assigned to separate GPUs provided that their output
directories do not overlap.

## Aggregate the paired runs

From the repository root, write the per-seed table, method summary, paired
differences, statistical report, and output hashes with:

```bash
python manuscript/Scripts/build_v60_gamma005_holdout.py \
  --result-root code/experiments/active_learning/results/main_evaluation \
  --output-dir manuscript/SourceData/Gamma005HoldoutAnalysis
```

The aggregator requires all 20 runs to be complete and verifies the method,
seed, frozen-protocol hash, MC-dropout count, `gamma`, trajectory length, and
within-method sequence uniqueness before writing results.

Rebuild the main held-out figure and the complete manuscript figure set with:

```bash
python manuscript/Scripts/rebuild_v69_holdout_figure.py
python scripts/validate_repository.py --figures
```

## Rebuild the archived benchmark figures

The six-policy benchmark and the archived Gate--Greedy evidence figure are
computed from the retained histories and bounded source tables:

```bash
python manuscript/Scripts/build_v50_recovery_figures.py
```

This produces the target-recovery curves, post-selection evaluability overlay,
and stopping-budget AUTC panels used in the manuscript.

## DFT boundary

The repository contains preparation and audit code, final candidate CIFs, and
bounded numerical summaries. It does not redistribute VASP executables,
POTCAR/PAW datasets, or large raw VASP outputs. Re-running first-principles
calculations requires a valid VASP license and locally supplied potentials.

## Citation and licensing

Citation metadata are provided in `CITATION.cff`. Source code is licensed under
the MIT License in `LICENSE`. Except where noted otherwise, data tables, source
data, figures, manuscript materials, and documentation are licensed under CC BY
4.0 as described in `DATA_LICENSE.md`.
