"""Independent Decimal implementation of the frozen formation-energy formula."""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Mapping

getcontext().prec = 40


def recompute_formation_energy_decimal(row: Mapping[str, object]) -> float:
    candidate = Decimal(str(row["candidate_total_energy_eV"]))
    lithium = Decimal(str(row["Li_energy_per_atom_eV"]))
    chromium = Decimal(str(row["Cr_energy_per_atom_eV"]))
    oxygen_molecule = Decimal(str(row["O2_energy_per_molecule_eV"]))
    value = (
        candidate
        - lithium
        - Decimal(2) * chromium
        - Decimal(2) * oxygen_molecule
    ) / Decimal(7)
    return float(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with Path(args.input).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["formation_energy_decimal_eV_atom"] = (
            f"{recompute_formation_energy_decimal(row):.12f}"
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
