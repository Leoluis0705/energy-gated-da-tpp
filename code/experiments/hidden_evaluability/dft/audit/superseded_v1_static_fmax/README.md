# Superseded audit output v1

These five CSV files are the first audit-script output generated on 2026-07-16. They are retained byte-for-byte and are not scientific source data.

The first `structure_metrics.csv` used forces from the final static OUTCAR for every candidate. That definition is unsuitable as the primary relaxation `Fmax` when a stage-specific relaxation OUTCAR exists, and it can be especially misleading for the two electronically unconverged new12 static jobs. The current required output instead reports relaxation-stage `Fmax` when available, retains static-stage `Fmax` in a separate column, and explicitly falls back to static forces for the eight pilots whose original relaxation OUTCAR files are unavailable. The Mg PAW label parser was also normalized to remove an evidence-file prefix.

No source VASP file or historical result was changed.

| File | Bytes | SHA-256 |
|---|---:|---|
| `convergence_inventory.csv` | 39,114 | `142c5b436823c399cb50153b33e8822786d6ac6a52c53587a543c437540b5133` |
| `dft_settings.csv` | 104,001 | `72d5e73a4e4af511d27e327f78dc1b1d747d21cb6efbec8d505e2f6a354c6b66` |
| `elemental_references.csv` | 5,506 | `cf5dfd852426badf190753075add1e80d232c578a0d3aef823bf3c956d0941f3` |
| `magnetic_initializations.csv` | 5,212 | `e8fe363672374a9b1c8738afd239e67d249fddc4f2d8dfb5808bab10e4e33422` |
| `structure_metrics.csv` | 15,414 | `940fdef5ae4c35e60676109845f9c3f22cd14b914e14177539d1f2f9a414c06a` |
