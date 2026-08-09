"""Primary self-consistent LiCr2O4 formation-energy calculation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Mapping


def compute_formation_energy(row: Mapping[str, object]) -> float:
    """Return LiCr2O4 formation energy in eV/atom.

    The O reference column is the energy of one O2 molecule.  The frozen
    composition therefore consumes two O2 molecules per LiCr2O4 formula unit.
    """

    candidate = float(row["candidate_total_energy_eV"])
    lithium = float(row["Li_energy_per_atom_eV"])
    chromium = float(row["Cr_energy_per_atom_eV"])
    oxygen_molecule = float(row["O2_energy_per_molecule_eV"])
    return (
        candidate - lithium - 2.0 * chromium - 2.0 * oxygen_molecule
    ) / 7.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with Path(args.input).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["formation_energy_primary_eV_atom"] = (
            f"{compute_formation_energy(row):.12f}"
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
