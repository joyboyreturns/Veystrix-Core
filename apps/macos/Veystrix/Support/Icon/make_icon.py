#!/usr/bin/env python3
"""Generate AppIcon.icns (original artwork) with Pillow + iconutil. Usage: python3 make_icon.py <out.icns>"""
import subprocess, sys, tempfile
from pathlib import Path
from PIL import Image, ImageDraw

def draw(size: int) -> Image.Image:
    s = size * 4                                       # supersample
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    r = int(s * 0.225)
    pad = int(s * 0.06)
    # vertical gradient background
    grad = Image.new("RGBA", (s, s))
    gd = ImageDraw.Draw(grad)
    for y in range(s):
        t = y / s
        gd.line([(0, y), (s, y)], fill=(int(20 + 30 * t), int(24 + 60 * t), int(60 + 110 * t), 255))
    mask = Image.new("L", (s, s), 0)
    ImageDraw.Draw(mask).rounded_rectangle([pad, pad, s - pad, s - pad], radius=r, fill=255)
    img.paste(grad, (0, 0), mask)
    # the V: two bold strokes meeting at the bottom, plus a thin streaming line
    w = int(s * 0.085)
    left, right, bottom, top = int(s * 0.28), int(s * 0.72), int(s * 0.72), int(s * 0.28)
    mid = s // 2
    d.line([(left, top), (mid, bottom)], fill=(255, 255, 255, 255), width=w, joint="curve")
    d.line([(right, top), (mid, bottom)], fill=(120, 200, 255, 255), width=w, joint="curve")
    for p in ((left, top), (mid, bottom), (right, top)):
        d.ellipse([p[0] - w // 2, p[1] - w // 2, p[0] + w // 2, p[1] + w // 2], fill=(255, 255, 255, 255) if p != (right, top) else (120, 200, 255, 255))
    y = int(s * 0.80)
    d.line([(int(s * 0.30), y), (int(s * 0.70), y)], fill=(255, 255, 255, 110), width=max(2, w // 4))
    return img.resize((size, size), Image.LANCZOS)

def main(out: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        iconset = Path(tmp) / "AppIcon.iconset"
        iconset.mkdir()
        for base in (16, 32, 128, 256, 512):
            draw(base).save(iconset / f"icon_{base}x{base}.png")
            draw(base * 2).save(iconset / f"icon_{base}x{base}@2x.png")
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", out], check=True)

if __name__ == "__main__":
    main(sys.argv[1])
