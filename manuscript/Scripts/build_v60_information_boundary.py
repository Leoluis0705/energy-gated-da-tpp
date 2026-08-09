"""Rebuild the information-boundary schematic as an editable vector figure."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


BLUE = "#2f6f9f"
BLUE_FILL = "#f3f8fc"
ORANGE = "#e1812c"
ORANGE_FILL = "#fff8f0"
GREEN = "#4c956c"
GREEN_FILL = "#f1f8f3"
PURPLE = "#756bb1"
PURPLE_FILL = "#f5f3fa"
GREY = "#666666"


def rounded(ax, x, y, w, h, title, lines, edge=BLUE, fill="white", title_size=9.5):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        linewidth=1.1,
        edgecolor=edge,
        facecolor=fill,
    )
    ax.add_patch(patch)
    title_lines = title.count("\n") + 1
    ax.text(x + w / 2, y + h - 0.022, title, ha="center", va="top", fontsize=title_size, weight="bold", color="#202020", linespacing=1.0)
    body = "\n".join(lines)
    body_y = y + h * (0.34 if title_lines > 1 else 0.40)
    ax.text(x + w / 2, body_y, body, ha="center", va="center", fontsize=8.8, linespacing=1.22, color="#202020")
    return patch


def arrow(ax, start, end, colour="#202020", style="-", width=1.2, mutation=11, connection="arc3"):
    patch = FancyArrowPatch(
        start, end,
        arrowstyle="-|>",
        mutation_scale=mutation,
        linewidth=width,
        linestyle=style,
        color=colour,
        connectionstyle=connection,
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(patch)
    return patch


def build(output_stem: Path) -> list[Path]:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.4,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, ax = plt.subplots(figsize=(7.4, 4.45))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    top = FancyBboxPatch((0.01, 0.63), 0.98, 0.34, boxstyle="round,pad=0.008,rounding_size=0.015", linewidth=1.3, edgecolor=BLUE, facecolor=BLUE_FILL)
    bottom = FancyBboxPatch((0.01, 0.075), 0.98, 0.51, boxstyle="round,pad=0.008,rounding_size=0.015", linewidth=1.3, edgecolor=ORANGE, facecolor=ORANGE_FILL)
    ax.add_patch(top)
    ax.add_patch(bottom)
    ax.text(0.025, 0.945, "Initialization and frozen task definition", fontsize=11.5, weight="bold", color=BLUE, va="top")
    ax.text(0.025, 0.555, "Query strategy", fontsize=11.5, weight="bold", color=ORANGE, va="top")

    rounded(ax, 0.025, 0.69, 0.18, 0.19, "Hidden labels", ["ALIGNN / MP", "revealed on query"], BLUE, "white")
    rounded(ax, 0.235, 0.69, 0.20, 0.19, "CGCNN prior", ["pretrained model", "shared by policies"], BLUE, "white")
    rounded(ax, 0.465, 0.69, 0.25, 0.19, "Frozen task", [r"$N$ candidates", r"target $[L,U]$", r"batch $b$"], BLUE, "white")
    rounded(ax, 0.745, 0.69, 0.23, 0.19, "Acquisition signals", [r"$p_i$: interval hit", r"$M_t$: margin", r"$G_t$: group share"], BLUE, "white", 9.2)

    rounded(ax, 0.025, 0.29, 0.12, 0.16, r"Pool $R_t$", ["unqueried IDs"], ORANGE, "white")
    rounded(ax, 0.175, 0.24, 0.25, 0.24, "Surrogate scoring", [r"$\mu_i,\sigma_i,\phi_i,g_i$", r"$p_i=P(L\leq y_i\leq U)$", r"compute $M_t,G_t$"], BLUE, "white")
    rounded(ax, 0.46, 0.22, 0.23, 0.28, "Conditional Gate", [r"pass: Greedy top-$b$", "fail: DA-TPP repair", "uncertainty + diversity"], PURPLE, PURPLE_FILL)
    rounded(ax, 0.72, 0.29, 0.115, 0.14, r"Batch $B_t$", ["selected IDs"], ORANGE, "white", 9.1)
    rounded(ax, 0.865, 0.33, 0.105, 0.16, "Oracle", [r"reveal $y_i$"], BLUE, "white", 9.2)
    rounded(ax, 0.69, 0.105, 0.145, 0.12, "Update + refit", ["queried labels"], PURPLE, PURPLE_FILL, 9.0)
    rounded(ax, 0.86, 0.105, 0.11, 0.14, "Recovery", [r"$I(q)$; AUTC"], GREEN, GREEN_FILL, 9.1)

    arrow(ax, (0.145, 0.36), (0.175, 0.36))
    arrow(ax, (0.425, 0.36), (0.46, 0.36))
    arrow(ax, (0.69, 0.36), (0.72, 0.36))
    arrow(ax, (0.835, 0.36), (0.865, 0.40))
    arrow(ax, (0.918, 0.33), (0.915, 0.245))
    arrow(ax, (0.882, 0.33), (0.80, 0.225))
    arrow(ax, (0.69, 0.16), (0.30, 0.24), colour=GREY, width=1.1, connection="arc3,rad=-0.18")

    arrow(ax, (0.335, 0.69), (0.30, 0.48), colour=BLUE, width=1.0)
    arrow(ax, (0.59, 0.69), (0.58, 0.50), colour=BLUE, width=1.0)
    arrow(ax, (0.105, 0.69), (0.895, 0.49), colour=GREY, style="--", width=1.0, connection="arc3,rad=0.08")
    ax.text(0.53, 0.61, "labels remain hidden until the oracle query", ha="center", va="center", fontsize=9.0, color=GREY, bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0})

    y = 0.028
    arrow(ax, (0.03, y), (0.08, y), width=1.1)
    ax.text(0.09, y, "selection flow", va="center", fontsize=9.0)
    arrow(ax, (0.26, y), (0.31, y), colour=GREY, style="--", width=1.0)
    ax.text(0.32, y, "hidden before query", va="center", fontsize=9.0)
    arrow(ax, (0.52, y), (0.57, y), colour=GREY, width=1.0)
    ax.text(0.58, y, "model-update feedback", va="center", fontsize=9.0)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for suffix in ("pdf", "svg", "png", "tiff"):
        dpi = 600 if suffix in {"png", "tiff"} else None
        output = output_stem.with_suffix(f".{suffix}")
        fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
        outputs.append(output)
    plt.close(fig)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-stem", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    outputs = build(args.output_stem)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "sha256", "bytes"))
        writer.writeheader()
        for output in outputs:
            writer.writerow(
                {
                    "path": str(output),
                    "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                    "bytes": output.stat().st_size,
                }
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
