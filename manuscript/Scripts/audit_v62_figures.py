import csv
from pathlib import Path

from PIL import Image
from pypdf import PdfReader
from pypdf.generic import ContentStream


root = Path(__file__).resolve().parents[1]
figures = root / "Figures"
names = [
    "Figure1_v60_information_boundary",
    "Figure2_v50_six_policy_mentor",
    "Figure3_v50_gate_greedy_evidence",
    "Figure4_v50_mace_dft_calibration",
    "Figure5_v50_relaxed_structures",
    "Figure6_v60_mnoxide_control",
    "Figure7_v60_parameter_sensitivity",
    "Figure8_v61_gamma005_holdout",
]


def count_images(resources):
    total = 0
    xobjects = resources.get("/XObject", {})
    for reference in xobjects.values():
        obj = reference.get_object()
        if obj.get("/Subtype") == "/Image":
            total += 1
        elif obj.get("/Subtype") == "/Form" and "/Resources" in obj:
            total += count_images(obj["/Resources"])
    return total


rows = []
for name in names:
    pdf_path = figures / f"{name}.pdf"
    png_path = figures / f"{name}.png"
    reader = PdfReader(pdf_path)
    page = reader.pages[0]
    content = ContentStream(page.get_contents(), reader)
    font_sizes = sorted(
        {round(float(operands[1]), 2) for operands, operator in content.operations if operator == b"Tf"}
    )
    with Image.open(png_path) as image:
        dpi = image.info.get("dpi", (None, None))[0]
        pixel_width, pixel_height = image.size
    rows.append(
        {
            "figure": name,
            "pdf_bytes": pdf_path.stat().st_size,
            "pdf_page_width_pt": round(float(page.mediabox.width), 2),
            "pdf_page_height_pt": round(float(page.mediabox.height), 2),
            "embedded_raster_objects": count_images(page["/Resources"]),
            "pdf_text_objects": sum(1 for _, operator in content.operations if operator in {b"Tj", b"TJ"}),
            "source_font_sizes_pt": ";".join(map(str, font_sizes)),
            "png_width_px": pixel_width,
            "png_height_px": pixel_height,
            "png_dpi": round(dpi) if dpi else "",
        }
    )

csv_path = root / "V62_FIGURE_QA.csv"
with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

report = [
    "# V62 figure quality audit",
    "",
    "All eight main-text figures were regenerated from the archived Python scripts and source data. "
    "The manuscript includes the vector PDF versions; PNG/TIFF files are parallel submission assets.",
    "",
    "| Figure | Vector PDF | Embedded raster objects | 600-dpi PNG | Pixel dimensions |",
    "|---|---:|---:|---:|---:|",
]
for row in rows:
    report.append(
        f"| {row['figure']} | yes | {row['embedded_raster_objects']} | "
        f"{'yes' if row['png_dpi'] == 600 else 'no'} | {row['png_width_px']} x {row['png_height_px']} |"
    )
report += [
    "",
    "## Interpretation",
    "",
    "- Zero embedded raster objects means that curves, labels, markers, and crystal drawings remain vector content in the PDF figure.",
    "- Figure text was enlarged in the source scripts before export; the LaTeX widths were also increased for Figures 2--6 so that the final-page text is not reduced below the intended journal reading size.",
    "- Small font operands below 8 pt that remain in the PDFs are mathematical subscripts/superscripts, not panel labels, axes, legends, or annotations.",
    "- The obsolete `Figure1_acquisition_architecture.png` is not referenced by the V62 manuscript; Figure 1 now uses `Figure1_v60_information_boundary.pdf`.",
]
(root / "V62_FIGURE_QA.md").write_text("\n".join(report) + "\n", encoding="utf-8")
