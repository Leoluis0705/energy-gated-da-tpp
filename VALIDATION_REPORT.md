# Validation report

Validation date: 10 August 2026

## Scientific code tests

The following tests were rerun from the clean staging repository:

| Component | Result | Runtime |
|---|---:|---:|
| Active-learning formal protocol, MC-dropout policy, uncertainty units, paired runner | 36 passed | 14.01 s |
| Hidden DFT-evaluability data/model/replay pipeline | 26 passed | 113.64 s |
| Prospective Cr DFT preparation/result audit | 11 passed | 9.94 s |

The active-learning group emitted one non-failing PyTorch warning about the
deprecated `pynvml` package.

## Figure reconstruction

The following v71 evidence was rebuilt successfully from the retained source
data and final CIFs:

- six-policy and Gate--Greedy recovery figures;
- MACE-MP leave-one-out calibration and MLIP runtime figure;
- four-candidate relaxed-structure figure;
- parameter-sensitivity figure;
- independently initialized held-out comparison.

## Repository security

The staging repository was scanned for known project passwords, GitHub token
formats, SSH private-key headers, POTCAR, WAVECAR, CHGCAR, OUTCAR, and
`vasprun.xml`. No credential or restricted-file matches were found. Historical
scripts with machine-specific path defaults remain as provenance, but the
documented validation and figure-rebuild entry points use repository-relative
paths.

## LaTeX

The v71 source includes the CMC class, definitions, references, figures, tables,
and body files. Fresh local pdfLaTeX compilation completed on 10 August 2026 for
the main manuscript (17 pages), Supplementary Material (11 pages), and cover
letter (1 page). Each PDF was rendered to raster previews for visual inspection;
no clipping, overlap, missing figure, or broken glyph was observed. Remaining
compiler messages are non-fatal template/PDF-string and underfull-box warnings.

## Current publication gates

- Authors must approve the public repository visibility.
- Authors or the institution must approve code and data licenses.
- GitHub authentication must be renewed before pushing.
- A Zenodo DOI must be minted from the tagged release before it is inserted in
  the manuscript.
