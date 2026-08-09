# Code inventory

## Active learning

The formal-run snapshot contains:

- `active_learning_energy_gate_ablation.py`: Energy-Gated DA-TPP, Greedy,
  Always-DA-TPP, margin-only, and group-only selection modes.
- `active_learning_etdg_tage.py`: archived selector implementation.
- `main.py`, `cgcnn/`: CGCNN training, prediction, and data loading.
- `mc_dropout_protocol.py`, `mc_dropout_seed_policy.py`: stochastic inference
  and deterministic mask-seed policy.
- `uncertainty_units.py`: conversion of predictive mean and standard deviation
  to physical units.
- `experiments/reproducibility/`: paired-run orchestration and audit artifacts.
- `analysis/`: aggregation, statistical recomputation, seed checks,
  calibration, and figure generation.

The candidate pool, reference-label table, atom embeddings, and project
checkpoint are included because the formal runner checks their identifiers and
hashes before execution.

## Hidden DFT-workflow evaluator

`analysis/three_system/` contains data assembly, shallow-model training,
leave-one-out validation, replay, and hidden post-selection scoring. The
retained inputs distinguish out-of-fold probabilities for historical DFT
examples from full-model probabilities assigned to the remaining pool.

## DFT assessment

`experiments/dft_audit/analysis/` contains the committed code for:

- input and k-point preparation;
- candidate and elemental-reference calculations;
- server-side job control;
- convergence and structure extraction;
- formation-energy reconstruction;
- prospective LiCr2O4 result collection.

The accompanying tests are retained under `experiments/dft_audit/tests/`.

## Manuscript evidence

`paper_rebuild/Scripts/` rebuilds the acquisition architecture, six-policy
trajectory plot, Gate--Greedy post-selection plot, DFT summary, and tabulated
evidence used in the submission.

