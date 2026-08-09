from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pymatgen.core import Structure
from scipy.spatial import ConvexHull, QhullError


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "SourceData" / "Figure4_DFT_feasibility.csv"
STRUCTURE_DIR = ROOT / "SourceData" / "DFT_relaxed_structures"
OUTPUT = ROOT / "Figures" / "Figure4_DFT_feasibility"

TARGET_INTERVAL = (-2.18, -2.02)
CANDIDATE_FILES = {
    "job_079 gen_1": "job_079_gen_1_CONTCAR",
    "job_126 gen_0": "job_126_gen_0_CONTCAR",
    "job_196 gen_1": "job_196_gen_1_CONTCAR",
    "job_234 gen_3": "job_234_gen_3_CONTCAR",
}

ELEMENT_COLORS = {
    "Li": "#76B82A",
    "Cr": "#2C7FB8",
    "O": "#D84A3A",
}
ELEMENT_SIZES = {
    "Li": 34,
    "Cr": 48,
    "O": 25,
}


def cell_corners(lattice: np.ndarray) -> np.ndarray:
    a, b, c = lattice
    return np.array(
        [
            i * a + j * b + k * c
            for i in (0, 1)
            for j in (0, 1)
            for k in (0, 1)
        ]
    )


def draw_cell(ax: mpl.axes.Axes, lattice: np.ndarray) -> np.ndarray:
    corners = cell_corners(lattice)
    index = {(i, j, k): 4 * i + 2 * j + k for i in (0, 1) for j in (0, 1) for k in (0, 1)}
    for i, j, k in index:
        start = corners[index[(i, j, k)]]
        for axis in range(3):
            end_index = [i, j, k]
            if end_index[axis] == 1:
                continue
            end_index[axis] = 1
            end = corners[index[tuple(end_index)]]
            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                [start[2], end[2]],
                color="#8A8A8A",
                linewidth=0.65,
                alpha=0.75,
                zorder=1,
            )
    return corners


def unique_rows(values: list[np.ndarray], decimals: int = 5) -> np.ndarray:
    if not values:
        return np.empty((0, 3))
    array = np.asarray(values, dtype=float)
    _, indices = np.unique(np.round(array, decimals=decimals), axis=0, return_index=True)
    return array[np.sort(indices)]


def draw_coordination_polyhedra(
    ax: mpl.axes.Axes, structure: Structure
) -> np.ndarray:
    periodic_oxygen: list[np.ndarray] = []
    for site in structure:
        if site.specie.symbol != "Cr":
            continue
        neighbours = [
            neighbour
            for neighbour in structure.get_neighbors(site, 2.45)
            if neighbour.specie.symbol == "O"
        ]
        oxygen = unique_rows([np.asarray(neighbour.coords) for neighbour in neighbours])
        if oxygen.shape[0] < 4:
            continue
        periodic_oxygen.extend(oxygen)
        center = np.asarray(site.coords)
        for point in oxygen:
            ax.plot(
                [center[0], point[0]],
                [center[1], point[1]],
                [center[2], point[2]],
                color="#2C7FB8",
                linewidth=0.55,
                alpha=0.75,
                zorder=2,
            )
        try:
            hull = ConvexHull(oxygen)
        except QhullError:
            continue
        faces = [oxygen[simplex] for simplex in hull.simplices]
        polyhedron = Poly3DCollection(
            faces,
            facecolor="#8CC4EA",
            edgecolor="#2C7FB8",
            linewidth=0.45,
            alpha=0.22,
            zorder=1.5,
        )
        ax.add_collection3d(polyhedron)
    return unique_rows(periodic_oxygen)


def draw_structure(ax: mpl.axes.Axes, structure: Structure) -> None:
    corners = draw_cell(ax, np.asarray(structure.lattice.matrix))
    periodic_oxygen = draw_coordination_polyhedra(ax, structure)

    points_for_limits = [corners]
    for element in ("Li", "Cr", "O"):
        coordinates = np.array(
            [
                np.asarray(site.coords)
                for site in structure
                if site.specie.symbol == element
            ]
        )
        if coordinates.size == 0:
            continue
        points_for_limits.append(coordinates)
        ax.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            coordinates[:, 2],
            s=ELEMENT_SIZES[element],
            color=ELEMENT_COLORS[element],
            edgecolors="white",
            linewidths=0.35,
            depthshade=True,
            zorder=4,
        )

    if periodic_oxygen.size:
        points_for_limits.append(periodic_oxygen)
        ax.scatter(
            periodic_oxygen[:, 0],
            periodic_oxygen[:, 1],
            periodic_oxygen[:, 2],
            s=ELEMENT_SIZES["O"] * 0.85,
            color=ELEMENT_COLORS["O"],
            edgecolors="white",
            linewidths=0.3,
            depthshade=True,
            zorder=3,
        )

    limits = np.vstack(points_for_limits)
    center = 0.5 * (limits.min(axis=0) + limits.max(axis=0))
    half_width = 0.54 * np.ptp(limits, axis=0).max()
    ax.set_xlim(center[0] - half_width, center[0] + half_width)
    ax.set_ylim(center[1] - half_width, center[1] + half_width)
    ax.set_zlim(center[2] - half_width, center[2] + half_width)
    ax.set_box_aspect((1, 1, 1))
    ax.set_proj_type("ortho")
    ax.view_init(elev=20, azim=-58)
    ax.set_axis_off()


def standardize_orientation(structure: Structure) -> Structure:
    lattice = np.asarray(structure.lattice.matrix)
    first = lattice[0] / np.linalg.norm(lattice[0])
    second_raw = lattice[1] - np.dot(lattice[1], first) * first
    second = second_raw / np.linalg.norm(second_raw)
    third = np.cross(first, second)
    if np.dot(third, lattice[2]) < 0:
        third *= -1
    rotation = np.column_stack([first, second, third])
    rotated_lattice = lattice @ rotation
    return Structure(
        rotated_lattice,
        [site.specie for site in structure],
        structure.frac_coords,
        coords_are_cartesian=False,
    )


def main() -> None:
    data = pd.read_csv(SOURCE).set_index("candidate")
    if list(data.index) != list(CANDIDATE_FILES):
        raise ValueError("candidate order or identifiers do not match the fixed DFT table")
    energies = data["formation_energy_eV_per_atom"].astype(float)
    if not energies.between(*TARGET_INTERVAL, inclusive="both").all():
        raise ValueError("one or more DFT energies fall outside the archived interval")

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    figure = plt.figure(figsize=(7.2, 4.65))
    axes = [figure.add_subplot(2, 2, index + 1, projection="3d") for index in range(4)]

    for panel_index, (candidate, filename) in enumerate(CANDIDATE_FILES.items()):
        structure_path = STRUCTURE_DIR / filename
        structure = standardize_orientation(Structure.from_file(structure_path))
        draw_structure(axes[panel_index], structure)
        energy = data.loc[candidate, "formation_energy_eV_per_atom"]
        axes[panel_index].text2D(
            0.02,
            0.98,
            f"({chr(97 + panel_index)}) {candidate.replace(' ', ' / ')}",
            transform=axes[panel_index].transAxes,
            ha="left",
            va="top",
            fontsize=8.1,
            fontweight="bold",
        )
        axes[panel_index].text2D(
            0.5,
            0.05,
            rf"$E_f={energy:.4f}$ eV atom$^{{-1}}$",
            transform=axes[panel_index].transAxes,
            ha="center",
            va="bottom",
            fontsize=7.2,
        )

    legend = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=ELEMENT_COLORS[element],
            markeredgecolor="white",
            markersize=6,
            label=element,
        )
        for element in ("Li", "Cr", "O")
    ]
    figure.legend(
        handles=legend,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.015),
        handletextpad=0.35,
        columnspacing=1.1,
    )
    figure.subplots_adjust(
        left=0.015,
        right=0.985,
        top=0.985,
        bottom=0.105,
        wspace=0.015,
        hspace=0.015,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(OUTPUT.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(OUTPUT.with_suffix(".png"), dpi=400, bbox_inches="tight")
    figure.savefig(
        OUTPUT.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
