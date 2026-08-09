# Energy-Gated DA-TPP v71 submission package

This package contains the claim-calibrated CMC manuscript and Supplementary Information prepared on 2026-08-09.

## Primary files

- `Energy_Gated_DA_TPP_v71_manuscript.pdf`: compiled 17-page main manuscript.
- `Energy_Gated_DA_TPP_v71_manuscript.tex`: main LaTeX entry file.
- `manuscript_body_v71.tex`: main manuscript body.
- `Energy_Gated_DA_TPP_v71_supplementary.pdf`: compiled 11-page Supplementary Information.
- `Energy_Gated_DA_TPP_v71_supplementary.tex`: SI source.
- `Approved_Abstract_Source.docx`: author-approved abstract source; its text is unchanged in v71.
- `V71_LANGUAGE_AND_CLAIM_AUDIT.md`: redundancy, overclaim, and over-conservatism audit.
- `V62_REFERENCE_VERIFICATION_REPORT.md`: reference verification record.
- `V71_SHA256_MANIFEST.csv`: package file hashes.

## Supporting directories

- `Definitions/`: CMC class and template assets.
- `Figures/`: publication figures and available vector/raster source formats.
- `Tables/`: LaTeX table sources.
- `SourceData/`: figure and table source data.
- `Scripts/`: figure generation and audit scripts.
- `Structures/`: final candidate structures and provenance files.

## Build

Compile the main manuscript and SI from the package root with a LaTeX engine compatible with the supplied CMC class. Both documents were compiled twice under MiKTeX before packaging. The final logs contained no LaTeX errors, undefined references, undefined citations, or overfull boxes.

## Evidence interpretation

The held-out Gate--Greedy comparison supports an aggregate early-budget ordering advantage under the frozen protocol. The post-selection DFT-evaluability model is acquisition-blind and does not represent observed DFT outcomes. The four LiCr2O4 workflows provide first-principles feasibility evidence but are not presented as phase-stability or synthesis validation.
