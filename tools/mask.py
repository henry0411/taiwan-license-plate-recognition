"""Privacy masking for paper figures (host python3 + Pillow).

mask(img, keep=None, mosaic=[], blur_outside=True)
  keep    : (x1, y1, x2, y2) region left sharp (the vehicle); everything else is blurred + darkened
  mosaic  : list of (x1, y1, x2, y2) rectangles that are pixelated hard (plates, faces, timestamps)
"""
from PIL import Image, ImageDraw, ImageFilter


def pixelate(img, box, block=None):
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.width, x2), min(img.height, y2)
    if x2 <= x1 or y2 <= y1:
        return img
    region = img.crop((x1, y1, x2, y2))
    block = block or max(6, min(x2 - x1, y2 - y1) // 3)
    small = region.resize((max(1, (x2 - x1) // block), max(1, (y2 - y1) // block)), Image.BILINEAR)
    img.paste(small.resize(region.size, Image.NEAREST), (x1, y1))
    return img


def mask(img, keep=None, mosaic=(), blur_outside=True, blur_radius=18):
    img = img.convert("RGB").copy()
    if keep is not None and blur_outside:
        blurred = img.filter(ImageFilter.GaussianBlur(blur_radius))
        blurred = Image.blend(blurred, Image.new("RGB", img.size, (40, 40, 40)), 0.35)
        blurred.paste(img.crop(tuple(map(int, keep))), tuple(map(int, keep[:2])))
        img = blurred
    for b in mosaic:
        pixelate(img, b)
    return img


def label_box(img, box, text, color=(0, 200, 80), width=3, font=None):
    d = ImageDraw.Draw(img)
    d.rectangle(tuple(map(int, box)), outline=color, width=width)
    if text:
        d.text((int(box[0]) + 2, max(0, int(box[1]) - 22)), text, fill=color, font=font)
    return img
