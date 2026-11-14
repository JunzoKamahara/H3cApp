"""Contact sheets of the phase-0 size comparison.
usage: sheet.py <phase0 dir> <prompt name> <WxH>...
Writes <dir>/<name>/sheet_full.jpg (each image scaled to 720 px height) and
sheet_crop.jpg (the central 512x512 of each at native pixels)."""
import json, sys
from pathlib import Path
from PIL import Image, ImageDraw
root, name, sizes = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
def label(im, text):
    d = ImageDraw.Draw(im); d.rectangle([0, 0, 8 + 7 * len(text), 18], fill=(0, 0, 0)); d.text((4, 3), text, fill=(255, 255, 255)); return im
full, crop = [], []
for size in sizes:
    im = Image.open(root / name / size / "image.png").convert("RGB")
    w, h = im.size
    full.append(label(im.resize((round(w * 720 / h), 720), Image.LANCZOS), size))
    c = 512
    box = ((w - c) // 2, (h - c) // 2, (w + c) // 2, (h + c) // 2)
    crop.append(label(im.crop(box), size + " centre 512 px"))
for images, out in ((full, "sheet_full.jpg"), (crop, "sheet_crop.jpg")):
    sheet = Image.new("RGB", (sum(i.width for i in images) + 8 * (len(images) - 1), max(i.height for i in images)), (255, 255, 255))
    x = 0
    for i in images:
        sheet.paste(i, (x, 0)); x += i.width + 8
    sheet.save(root / name / out, quality=90)
    print(root / name / out, sheet.size)
