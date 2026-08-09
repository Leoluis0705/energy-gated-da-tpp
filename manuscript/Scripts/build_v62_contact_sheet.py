from pathlib import Path

from PIL import Image, ImageDraw


root = Path(__file__).resolve().parents[1]
pages = sorted((root / "qa_v62_final_all").glob("page-*.png"))
thumb_w = 306
margin = 18
label_h = 28
cols = 4

thumbs = []
for page in pages:
    with Image.open(page) as image:
        image = image.convert("RGB")
        thumb_h = round(image.height * thumb_w / image.width)
        thumbs.append((page.name, image.resize((thumb_w, thumb_h))))

rows = (len(thumbs) + cols - 1) // cols
cell_h = max(image.height for _, image in thumbs) + label_h
sheet = Image.new(
    "RGB",
    (margin + cols * (thumb_w + margin), margin + rows * (cell_h + margin)),
    "white",
)
draw = ImageDraw.Draw(sheet)
for index, (name, image) in enumerate(thumbs):
    row, col = divmod(index, cols)
    x = margin + col * (thumb_w + margin)
    y = margin + row * (cell_h + margin)
    sheet.paste(image, (x, y + label_h))
    draw.text((x, y + 5), name, fill="black")

sheet.save(root / "qa_v62_final_contact_sheet.png", dpi=(150, 150))
