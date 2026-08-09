# Superseded DFT extraction v2

This extraction is preserved for audit history. It must not be used as the current DFT audit table set.

In this version, every missing OUTCAR was assigned `original_relaxation_artifact_unavailable`. That wording is correct for the overwritten pilot relaxation outputs, but not for four unpopulated new-candidate GGA+U stage directories. The latter are now labeled `stage_output_unavailable_in_archived_job_bundle`; the audit does not infer that those stages were launched.

The correction changes only missing-output provenance/failure labels in `dft_settings.csv` and `convergence_inventory.csv`. The remaining tables are preserved here as a complete deterministic extraction set.

Original hashes:

- `dft_settings.csv`: `a470626f070801384fee98eed49dabc91c823d1ab89e0176ea779ea09c8d6640`
- `convergence_inventory.csv`: `142c5b436823c399cb50153b33e8822786d6ac6a52c53587a543c437540b5133`
- `structure_metrics.csv`: `49ceb41dd367064ad3195aaca2613aa0dadbb87bbaef8bd3b0680dfb7d5a0eca`
- `magnetic_initializations.csv`: `6f5dcdbade70932c8219a4d7aea7483d177459d589fb6696d7f4f78ff0625a83`
- `elemental_references.csv`: `ae56f66acd4b5ad6507eb2a4bf403f145a465e540b2d48df9808eed57236cf4b`
