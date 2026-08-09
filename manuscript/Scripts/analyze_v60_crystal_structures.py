from __future__ import annotations

import hashlib
from itertools import combinations
from pathlib import Path

import pandas as pd
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure


ROOT = Path(__file__).resolve().parents[1]
STRUCTURE_DIR = ROOT / "Structures"
SOURCE_DIR = ROOT / "SourceData"
TABLE_DIR = ROOT / "Tables" / "generated"
NAMES = ("C079-1", "C126-0", "C196-1", "C234-3")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_structures() -> dict[str, Structure]:
    structures = {}
    for name in NAMES:
        path = STRUCTURE_DIR / f"{name}_final.cif"
        if not path.is_file():
            raise FileNotFoundError(path)
        structure = Structure.from_file(path)
        if structure.composition.reduced_formula != "LiCr2O4" or len(structure) != 7:
            raise RuntimeError(f"Unexpected structure content in {path}")
        structures[name] = structure
    return structures


def structure_descriptors(structures: dict[str, Structure]) -> pd.DataFrame:
    rows = []
    for name, structure in structures.items():
        a, b, c = structure.lattice.abc
        alpha, beta, gamma = structure.lattice.angles
        spg_001, number_001 = structure.get_space_group_info(symprec=0.01)
        spg_005, number_005 = structure.get_space_group_info(symprec=0.05)
        spg_010, number_010 = structure.get_space_group_info(symprec=0.10)
        path = STRUCTURE_DIR / f"{name}_final.cif"
        rows.append(
            {
                "candidate": name,
                "formula": structure.composition.reduced_formula,
                "sites": len(structure),
                "volume_A3": structure.volume,
                "a_A": a,
                "b_A": b,
                "c_A": c,
                "alpha_deg": alpha,
                "beta_deg": beta,
                "gamma_deg": gamma,
                "space_group_symprec_0p01": spg_001,
                "space_group_number_symprec_0p01": number_001,
                "space_group_symprec_0p05": spg_005,
                "space_group_number_symprec_0p05": number_005,
                "space_group_symprec_0p10": spg_010,
                "space_group_number_symprec_0p10": number_010,
                "cif_sha256": sha256(path),
            }
        )
    return pd.DataFrame(rows)


def pairwise_matching(structures: dict[str, Structure]) -> pd.DataFrame:
    matcher = StructureMatcher(
        ltol=0.2,
        stol=0.3,
        angle_tol=5,
        primitive_cell=True,
        scale=True,
        attempt_supercell=False,
    )
    rows = []
    for first, second in combinations(NAMES, 2):
        is_match = matcher.fit(structures[first], structures[second])
        rms = matcher.get_rms_dist(structures[first], structures[second])
        rows.append(
            {
                "candidate_1": first,
                "candidate_2": second,
                "structure_match": bool(is_match),
                "normalized_rms": None if rms is None else rms[0],
                "max_distance": None if rms is None else rms[1],
                "ltol": 0.2,
                "stol": 0.3,
                "angle_tol_deg": 5,
            }
        )
    return pd.DataFrame(rows)


def write_table(frame: pd.DataFrame) -> None:
    lines = [
        r"\begin{tabular}{lrrrrll}",
        r"\toprule",
        r"Candidate & $a$ (\AA) & $b$ (\AA) & $c$ (\AA) & $V$ (\AA$^3$) & SG (0.01 \AA) & SG (0.10 \AA) \\",
        r"\midrule",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"{row.candidate} & {row.a_A:.3f} & {row.b_A:.3f} & {row.c_A:.3f} & "
            f"{row.volume_A3:.3f} & {row.space_group_symprec_0p01} & "
            f"{row.space_group_symprec_0p10} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    (TABLE_DIR / "table_v60_crystallography.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    structures = load_structures()
    descriptors = structure_descriptors(structures)
    matching = pairwise_matching(structures)
    descriptors.to_csv(
        SOURCE_DIR / "v60_crystallographic_descriptors.csv",
        index=False,
        float_format="%.10f",
    )
    matching.to_csv(
        SOURCE_DIR / "v60_structure_matcher_pairs.csv",
        index=False,
        float_format="%.10f",
    )
    write_table(descriptors)
    if matching["structure_match"].any():
        raise RuntimeError("At least one candidate pair matched unexpectedly")


if __name__ == "__main__":
    main()
