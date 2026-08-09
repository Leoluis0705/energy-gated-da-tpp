# Data dictionary

## Core active-learning data

- `code/experiments/active_learning/EARLY_LiM2O4_ALIGNN_HARD640_m2p18_m2p02_20260617/`:
  frozen Li--M--O pool and ALIGNN reference labels used by the main replay.
- `code/experiments/active_learning/NON_GEN_INTERVAL_POOLS_20260618/`:
  frozen Materials Project Mn-oxide control pool.
- `code/experiments/active_learning/checkpoint_formation_clean.pth.tar`:
  project CGCNN prior used by the archived workflow.
- `manuscript/SourceData/paired_cgcnn_histories.csv`:
  Gate--Greedy histories used in the main recovery evidence.
- `manuscript/SourceData/Gamma005HoldoutAnalysis/`:
  independently initialized held-out seeds 15--24 and paired statistics.

## DFT-evaluability data

- `manuscript/SourceData/historical_dft_binary_labels.csv`:
  bounded historical workflow outcomes used by the exploratory evaluator.
- `manuscript/SourceData/dft_evaluability_model_cv.csv`:
  nested leave-one-out model assessment.
- `manuscript/SourceData/dft_evaluability_scores.csv`:
  acquisition-blind post-selection scores.

These scores are diagnostic predictions, not additional DFT calculations.

## MLIP and DFT assessment data

- `manuscript/SourceData/mlip_full_pool_results.csv`:
  CHGNet and MACE-MP full-pool inference/runtime results.
- `manuscript/SourceData/energy_calibration_best_oof.csv`:
  leave-one-out MACE-MP/DFT calibration values.
- `manuscript/Structures/*_final.cif`:
  final structures of the four completed LiCr2O4 workflows shown in the main
  text.
- `manuscript/SourceData/v50_final_structure_provenance.csv`:
  bounded structural provenance and integrity hashes.

## Parameter and control evidence

- `manuscript/SourceData/v60_parameter_sensitivity_*.csv`:
  frozen development-grid sensitivity results.
- `manuscript/SourceData/mnoxide_*.csv`:
  control-pool trajectories, grouping inventory, and seed audit.
- `manuscript/SourceData/v60_same_protocol_*.csv`:
  same-protocol baseline comparisons.

