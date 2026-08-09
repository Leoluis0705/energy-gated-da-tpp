# Tested environment

Validation was performed on Windows with Python 3.11.9. Exact Python package
versions are listed in `requirements-lock.txt`; minimum supported versions are
listed in `requirements.txt` and `environment.yml`.

Complete active-learning trajectories require a CUDA-capable PyTorch build.
The checked environment used PyTorch 2.11.0 with CUDA 12.8, although no GPU is
required for the repository test suite or figure rebuild.

The formal protocol controls Python, NumPy, PyTorch, MC-dropout, and candidate
ordering seeds. Within a paired seed, methods use the same initial labelled set
and stochastic-mask schedule.

For manuscript compilation, use a complete TeX Live installation with
`latexmk` and run from the `manuscript/` directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error Energy_Gated_DA_TPP_v71_manuscript.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error Energy_Gated_DA_TPP_v71_supplementary.tex
```

The locally installed MiKTeX instance was not initialized during repository
validation. Bundled Tectonic reached the CMC class but failed because the class
selects a font backend inconsistent with Tectonic's engine. Precompiled v71
PDFs are retained for exact visual reference.

