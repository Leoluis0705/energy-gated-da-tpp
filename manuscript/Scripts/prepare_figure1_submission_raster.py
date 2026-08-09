"""Create publication-export copies of the user-supplied Figure 1 without altering content."""

from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Figures" / "Figure1_user_supplied_workflow.png"
PNG_OUT = ROOT / "Figures" / "Figure1_user_supplied_workflow_600dpi.png"
TIFF_OUT = ROOT / "Figures" / "Figure1_user_supplied_workflow_600dpi.tif"


def main() -> None:
    with Image.open(SOURCE).convert("RGB") as image:
        enlarged = image.resize((image.width * 2, image.height * 2), Image.Resampling.LANCZOS)
        enlarged.save(PNG_OUT, dpi=(600, 600), optimize=True)
        enlarged.save(TIFF_OUT, dpi=(600, 600), compression="tiff_lzw")


if __name__ == "__main__":
    main()
