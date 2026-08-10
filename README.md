# Energy-Gated DA-TPP reproducibility repository

This repository accompanies the manuscript:

> **Energy-Gated DA-TPP: Batch Active Learning for Early Target Recovery in Generated Crystal Pools**

It contains the frozen data, exact experimental code snapshots, manuscript and
Supplementary Material sources, figure/table builders, and audit records needed
to reproduce the reported results. The DFT-evaluability model is a
post-selection diagnostic and is not used by the acquisition policy.

## Recommended archival route

The maintained source should be hosted on GitHub. Each manuscript-linked
release should then be archived through Zenodo so that the cited version has an
immutable DOI. The DOI and release URL must be added to the manuscript only
after Zenodo has minted them.

## Repository map

- `code/experiments/active_learning/` -- CGCNN training, acquisition policies,
  formal protocol, calibration, analysis, frozen pools, checkpoint, and tests.
- `code/experiments/hidden_evaluability/` -- acquisition-blind DFT workflow
  evaluability model, leave-one-out analysis, replay, scores, and tests.
- `code/experiments/dft_audit/` -- VASP input preparation and post-processing,
  convergence/structure checks, formation-energy reconstruction, and tests.
- `data/` -- bounded source tables and archived histories required by the
  analyses.
- `manuscript/` -- current v71 CMC main text, Supplementary Material, figures,
  tables, source data, final CIFs, and figure builders.
- `scripts/validate_repository.py` -- one-command integrity, test, and figure
  rebuild entry point.
- `provenance/` -- source and public-file SHA-256 manifests plus security audit.

## Quick start

Python 3.11 is the tested interpreter.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/validate_repository.py --tests --figures
```

GPU reproduction of complete active-learning trajectories requires a CUDA build
of PyTorch. Tests, evidence-table checks, DFT post-processing, and manuscript
figure rebuilding are CPU-compatible.

## Reproduce the formal active-learning protocol

From `code/experiments/active_learning`:

```bash
python experiments/reproducibility/run_paired_dataset_job.py \
  --project-root . \
  --dataset limo \
  --method energy_gated_da_tpp \
  --seed 15 \
  --run-dir results/reproduction/limo/energy_gated_da_tpp/seed_15 \
  --protocol-config configs/frozen_final_protocol.yaml
```

The frozen protocol permits the methods and seeds declared in
`configs/frozen_final_protocol.yaml`. A full trajectory retrains the CGCNN and
is intended for a GPU environment.

## Rebuild manuscript figures

The repository validator rebuilds the figures used in the v71 manuscript from
`manuscript/SourceData/` and `manuscript/Structures/`. Individual builders are
also available under `manuscript/Scripts/`.

## DFT boundary

The repository contains preparation and audit code, final candidate CIFs, and
bounded numerical summaries. It does not redistribute VASP executables,
POTCAR/PAW datasets, or large raw VASP outputs. Re-running first-principles
calculations requires a valid VASP license and locally supplied potentials.

## Integrity and validation

Fresh validation completed on 10 August 2026 with:

- 36 active-learning protocol tests passed;
- 26 hidden-evaluability tests passed;
- 11 DFT audit/post-processing tests passed;
- manuscript figures rebuilt from retained source data.

See `VALIDATION_REPORT.md` for commands and environment details. The current
CMC source was compiled with MiKTeX pdfLaTeX on 10 August 2026. The final main
manuscript, Supplementary Material, and cover letter compiled without fatal
errors and were visually checked after rendering (17, 11, and 1 pages,
respectively).

## Citation and licensing

Citation metadata are provided in `CITATION.cff`. Source code is licensed under
the MIT License in `LICENSE`. Except where noted otherwise, data tables, source
data, figures, manuscript materials, and documentation are licensed under
CC BY 4.0 as described in `DATA_LICENSE.md`. VASP executables, POTCAR/PAW
datasets, and other license-restricted materials are not distributed.
