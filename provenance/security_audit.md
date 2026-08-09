# Public-repository security audit

Audit date: 10 August 2026

## Results

- No known project passwords were found in text-searchable files.
- No GitHub personal-access-token patterns were found.
- No SSH private-key headers were found.
- No `POTCAR`, `WAVECAR`, `CHGCAR`, `OUTCAR`, or `vasprun.xml` files were
  included.
- No file exceeds GitHub's 100 MB per-file limit.

Thirty-seven Python or shell scripts contain historical machine-specific path
examples or remote orchestration defaults. These scripts contain no embedded
credentials. They are retained for provenance, while the documented validation
and figure-rebuild paths are repository-relative. Local credentials and paths
must be supplied through ignored local configuration or environment variables.

## Publication rule

Run the following before every public release:

```bash
python scripts/validate_repository.py --tests --figures --manifest
```

Then review `provenance/public_manifest_sha256.csv` and the staged Git file list.

