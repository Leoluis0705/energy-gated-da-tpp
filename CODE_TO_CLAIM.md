# Code-to-claim map

| Manuscript evidence | Frozen source | Rebuild or audit code |
|---|---|---|
| Six-policy recovery histories | `manuscript/SourceData/Figure2_six_policy_target_recovery.csv` | `manuscript/Scripts/build_v50_recovery_figures.py` |
| Gate--Greedy early recovery | `manuscript/SourceData/paired_cgcnn_histories.csv` | `manuscript/Scripts/build_v50_recovery_figures.py` |
| Hidden DFT-evaluability audit | `manuscript/SourceData/dft_evaluability_*.csv` | `code/experiments/hidden_evaluability/analysis/three_system/` |
| MACE-MP/DFT calibration | `manuscript/SourceData/energy_calibration_best_oof.csv` | `manuscript/Scripts/build_v50_mlip_figures.py` |
| Four relaxed LiCr2O4 structures | `manuscript/Structures/` | `manuscript/Scripts/build_v50_relaxed_structures.py` |
| Parameter sensitivity | `manuscript/SourceData/v60_parameter_sensitivity_*.csv` | `manuscript/Scripts/rebuild_v63_discussion_figures.py` |
| Held-out seeds 15--24 | `manuscript/SourceData/Gamma005HoldoutAnalysis/` | `manuscript/Scripts/rebuild_v69_holdout_figure.py` |
| Formal acquisition implementation | frozen pools/config/checkpoint under `code/experiments/active_learning/` | `code/experiments/active_learning/experiments/reproducibility/` |
| DFT settings and bounded result audit | DFT tables and CIFs under `manuscript/` | `code/experiments/dft_audit/analysis/` |

