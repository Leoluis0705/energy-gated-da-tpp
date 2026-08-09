from __future__ import annotations

import hashlib
import os
from pathlib import Path
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter
from pymatgen.io.vasp.outputs import Vasprun


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "Figures"
SOURCE_DATA = ROOT / "SourceData"
STRUCTURES = ROOT / "Structures"

RAW_ROOT = Path(
    os.environ.get("V50_VASP_ARCHIVE", ROOT / "archived_vasp_records")
)

RECORDS = {
    "C079-1": RAW_ROOT
    / "job_079_Cr_fe_-0.854_n4_generated_crystals_cif__gen_1__static_0p15"
    / "attempt_0001"
    / "vasprun.xml",
    "C126-0": RAW_ROOT
    / "job_126_Cr_fe_-0.901_n4_generated_crystals_cif__gen_0__static_0p15"
    / "attempt_0001"
    / "vasprun.xml",
    "C196-1": RAW_ROOT
    / "job_196_Cr_fe_-0.819_n4_generated_crystals_cif__gen_1__static_0p15"
    / "attempt_0002"
    / "vasprun.xml",
    "C234-3": RAW_ROOT
    / "job_234_Cr_fe_-1.123_n4_generated_crystals_cif__gen_3__static_0p15"
    / "attempt_0002"
    / "vasprun.xml",
}

ELEMENT_STYLE = {
    "Li": dict(color="#B8BDC6", size=92, edge="#666B73"),
    "Cr": dict(color="#2EC4B6", size=245, edge="#137F78"),
    "O": dict(color="#92278F", size=112, edge="#5E155C"),
}

CELL_COLOR = "#3D4652"
BOND_STYLE = {
    ("Cr", "O"): dict(color="#37B9AD", cutoff=2.35, linewidth=1.45),
    ("Li", "O"): dict(color="#8B9098", cutoff=2.55, linewidth=1.05),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9.8,
            "axes.titlesize": 10.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "energy-gated-da-tpp-v50-structures",
        }
    )


def rotation_matrix() -> np.ndarray:
    x_angle = np.deg2rad(62.0)
    z_angle = np.deg2rad(-28.0)
    rotate_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(x_angle), -np.sin(x_angle)],
            [0, np.sin(x_angle), np.cos(x_angle)],
        ]
    )
    rotate_z = np.array(
        [
            [np.cos(z_angle), -np.sin(z_angle), 0],
            [np.sin(z_angle), np.cos(z_angle), 0],
            [0, 0, 1],
        ]
    )
    return rotate_x @ rotate_z


def project(coords: np.ndarray, center: np.ndarray) -> np.ndarray:
    rotated = (rotation_matrix() @ (coords - center).T).T
    return rotated


def cell_edges(lattice: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    corners = np.array(
        [
            i * lattice[0] + j * lattice[1] + k * lattice[2]
            for i in (0, 1)
            for j in (0, 1)
            for k in (0, 1)
        ]
    )
    indices = [(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)]
    edges: list[tuple[int, int]] = []
    for first, first_index in enumerate(indices):
        for second, second_index in enumerate(indices):
            if second <= first:
                continue
            if sum(a != b for a, b in zip(first_index, second_index, strict=True)) == 1:
                edges.append((first, second))
    return corners, edges


def parse_structures() -> tuple[dict[str, Structure], pd.DataFrame]:
    STRUCTURES.mkdir(parents=True, exist_ok=True)
    parsed: dict[str, Structure] = {}
    provenance: list[dict[str, str | float | int]] = []
    recorded_path = SOURCE_DATA / "v50_final_structure_provenance.csv"
    recorded = (
        pd.read_csv(recorded_path).set_index("candidate")
        if recorded_path.is_file()
        else pd.DataFrame()
    )
    for candidate, xml_path in RECORDS.items():
        cif_path = STRUCTURES / f"{candidate}_final.cif"
        if xml_path.is_file():
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="No POTCAR file with matching TITEL")
                run = Vasprun(
                    xml_path,
                    parse_dos=False,
                    parse_eigen=False,
                    exception_on_bad_xml=False,
                )
            structure = run.final_structure
            CifWriter(structure, symprec=None).write_file(cif_path)
            source_path = xml_path.as_posix()
            source_hash = sha256(xml_path)
        elif cif_path.is_file() and candidate in recorded.index:
            structure = Structure.from_file(cif_path)
            source_path = str(recorded.loc[candidate, "source_vasprun_xml"])
            source_hash = str(recorded.loc[candidate, "source_vasprun_sha256"])
        else:
            raise FileNotFoundError(
                f"Neither source vasprun.xml nor archived final CIF is available for {candidate}"
            )
        if structure.composition.reduced_formula != "LiCr2O4":
            raise AssertionError(
                f"{candidate} has unexpected composition {structure.composition}"
            )
        if len(structure) != 7:
            raise AssertionError(f"{candidate} has {len(structure)} sites, expected 7")

        reread = Structure.from_file(cif_path)
        if reread.composition.reduced_formula != "LiCr2O4":
            raise AssertionError(f"CIF composition mismatch for {candidate}")
        if len(reread) != len(structure):
            raise AssertionError(f"CIF site-count mismatch for {candidate}")
        if not np.isclose(reread.volume, structure.volume, rtol=1e-6, atol=1e-8):
            raise AssertionError(f"CIF volume mismatch for {candidate}")

        parsed[candidate] = structure
        a, b, c = structure.lattice.abc
        alpha, beta, gamma = structure.lattice.angles
        provenance.append(
            {
                "candidate": candidate,
                "source_vasprun_xml": source_path,
                "source_vasprun_sha256": source_hash,
                "final_cif": cif_path.relative_to(ROOT).as_posix(),
                "final_cif_sha256": sha256(cif_path),
                "reduced_formula": structure.composition.reduced_formula,
                "site_count": len(structure),
                "volume_A3": structure.volume,
                "a_A": a,
                "b_A": b,
                "c_A": c,
                "alpha_deg": alpha,
                "beta_deg": beta,
                "gamma_deg": gamma,
            }
        )
    frame = pd.DataFrame(provenance)
    frame.to_csv(
        SOURCE_DATA / "v50_final_structure_provenance.csv",
        index=False,
        float_format="%.10f",
    )
    return parsed, frame


def draw_structure(ax: mpl.axes.Axes, structure: Structure, title: str) -> None:
    lattice = np.asarray(structure.lattice.matrix)
    center = 0.5 * lattice.sum(axis=0)
    corners, edges = cell_edges(lattice)
    projected_corners = project(corners, center)

    # Bonds are defined by local coordination only; no bond order is implied.
    outside_atoms: list[tuple[str, np.ndarray]] = []
    for index, site in enumerate(structure):
        center_element = str(site.specie)
        for (first, second), style in BOND_STYLE.items():
            if center_element != first:
                continue
            for neighbor in structure.get_neighbors(site, style["cutoff"]):
                if str(neighbor.specie) != second:
                    continue
                segment = project(
                    np.vstack([site.coords, neighbor.coords]),
                    center,
                )
                depth = float(segment[:, 2].mean())
                alpha = 0.42 + 0.28 * (depth - projected_corners[:, 2].min()) / (
                    np.ptp(projected_corners[:, 2]) + 1e-9
                )
                ax.plot(
                    segment[:, 0],
                    segment[:, 1],
                    color=style["color"],
                    linewidth=style["linewidth"],
                    alpha=float(np.clip(alpha, 0.38, 0.72)),
                    zorder=2,
                )
                if tuple(int(round(value)) for value in neighbor.image) != (0, 0, 0):
                    outside_atoms.append((second, np.asarray(neighbor.coords)))

    # Draw the unit cell before atoms but above bonds.
    for first, second in edges:
        points = projected_corners[[first, second]]
        ax.plot(
            points[:, 0],
            points[:, 1],
            color=CELL_COLOR,
            linewidth=0.72,
            alpha=0.82,
            zorder=3,
        )

    atom_records: list[tuple[str, np.ndarray, bool]] = [
        (str(site.specie), np.asarray(site.coords), True) for site in structure
    ]
    seen_outside: set[tuple[str, float, float, float]] = set()
    for element, coords in outside_atoms:
        key = (element, *(round(float(value), 5) for value in coords))
        if key in seen_outside:
            continue
        seen_outside.add(key)
        atom_records.append((element, coords, False))

    projected_atoms = [
        (element, project(np.asarray([coords]), center)[0], inside)
        for element, coords, inside in atom_records
    ]
    projected_atoms.sort(key=lambda record: record[1][2])

    xmin, ymin = projected_corners[:, :2].min(axis=0)
    xmax, ymax = projected_corners[:, :2].max(axis=0)
    margin = 0.11 * max(xmax - xmin, ymax - ymin)
    for element, point, inside in projected_atoms:
        if not (
            xmin - margin <= point[0] <= xmax + margin
            and ymin - margin <= point[1] <= ymax + margin
        ):
            continue
        style = ELEMENT_STYLE[element]
        depth_fraction = (point[2] - projected_corners[:, 2].min()) / (
            np.ptp(projected_corners[:, 2]) + 1e-9
        )
        scale = 0.78 + 0.28 * float(np.clip(depth_fraction, 0, 1))
        ax.scatter(
            point[0],
            point[1],
            s=style["size"] * scale * (1.0 if inside else 0.74),
            c=style["color"],
            edgecolors=style["edge"],
            linewidths=0.65,
            alpha=1.0 if inside else 0.76,
            zorder=5 + depth_fraction,
        )
        ax.scatter(
            point[0] - 0.035 * margin,
            point[1] + 0.035 * margin,
            s=style["size"] * scale * 0.12,
            c="white",
            edgecolors="none",
            alpha=0.55 if inside else 0.30,
            zorder=6 + depth_fraction,
        )

    ax.set_xlim(xmin - margin, xmax + margin)
    ax.set_ylim(ymin - margin, ymax + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, loc="left", fontweight="bold", pad=2)
    ax.axis("off")


def save_all(fig: mpl.figure.Figure) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / "Figure5_v50_relaxed_structures"
    common = dict(bbox_inches="tight", pad_inches=0.035, facecolor="white")
    fig.savefig(
        path.with_suffix(".pdf"),
        **common,
        metadata={
            "Creator": "Energy-Gated DA-TPP v50 final-structure script",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(
        path.with_suffix(".svg"),
        **common,
        metadata={
            "Creator": "Energy-Gated DA-TPP v50 final-structure script",
            "Date": None,
        },
    )
    fig.savefig(
        path.with_suffix(".png"),
        dpi=600,
        **common,
        metadata={"Software": "Energy-Gated DA-TPP v50 final-structure script"},
    )
    fig.savefig(
        path.with_suffix(".tiff"),
        dpi=600,
        **common,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def draw_plate(structures: dict[str, Structure]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.05))
    labels = ("C079-1", "C126-0", "C196-1", "C234-3")
    for panel, (ax, label) in enumerate(zip(axes.flat, labels, strict=True)):
        draw_structure(ax, structures[label], f"{chr(97 + panel)}  {label}")

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=ELEMENT_STYLE[element]["color"],
            markeredgecolor=ELEMENT_STYLE[element]["edge"],
            markersize=7.5 if element == "Cr" else 5.8,
            label=element,
        )
        for element in ("Li", "Cr", "O")
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.012),
        columnspacing=1.5,
        handletextpad=0.45,
    )
    fig.subplots_adjust(left=0.025, right=0.99, top=0.99, bottom=0.075, wspace=0.04, hspace=0.10)
    save_all(fig)


def main() -> None:
    set_style()
    structures, provenance = parse_structures()
    draw_plate(structures)
    print(
        provenance[["candidate", "site_count", "volume_A3", "final_cif"]].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
