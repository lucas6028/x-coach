"""Derive the login page's brand-stage hero from the movement-library squat illustration.

Same kind of asset pipeline as `prep_movement_art.py`, and here for the same reason: the binary it
writes is committed, so the recipe that produced it has to be too.

Source: `frontend/public/movements/squat.png` -- the bodyweight-squat card art, an opaque 1254x1254
pre-matted square. The login stage needs three things the card does not:

  * a PURE white plate. The card's is (252,251,251), and the stage composites the hero with
    `mix-blend-multiply` so it sits on the violet floor gradient without an alpha halo; anything
    below 255 multiplies the gradient a shade darker and the plate shows as a faint rectangle.
  * a trim to the figure. The card square is ~35% padding, and the stage positions the art by
    percentage against the floor rings -- padding baked into the image floats the feet off them.
  * a much smaller file. 1.2 MB of PNG is not what an unauthenticated page should be spending.

Usage (from the repo root):
    .venv\\Scripts\\python.exe scripts/prep_login_hero.py
"""

import argparse
import os

import cv2
import numpy as np
from PIL import Image

# Colour distance from the SEED that still counts as plate. The background is a noisy near-white
# (250-253); 12 covers the noise and stops well short of the figure, whose lightest real pixels are
# the shoe highlights -- and those are interior, unreachable from the border seeds anyway.
TOL = 12
# Kept deliberately loose so the soft contact shadow under the feet survives the trim: it is what
# seats the figure on the stage's floor rings.
TRIM_THRESHOLD = 245
LONG_EDGE = 900


def main() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=os.path.join(root, "frontend", "public", "movements", "squat.png"))
    ap.add_argument("--dst", default=os.path.join(root, "frontend", "public", "assets", "squat-hero.webp"))
    args = ap.parse_args()

    rgb = np.asarray(Image.open(args.src).convert("RGB")).copy()
    h, w, _ = rgb.shape

    # Flood the plate to pure white, inwards from every border seed.
    mask = np.zeros((h + 2, w + 2), np.uint8)
    flags = 4 | (255 << 8) | cv2.FLOODFILL_FIXED_RANGE
    for sx, sy in (
        [(x, 0) for x in range(0, w, 4)]
        + [(x, h - 1) for x in range(0, w, 4)]
        + [(0, y) for y in range(0, h, 4)]
        + [(w - 1, y) for y in range(0, h, 4)]
    ):
        if mask[sy + 1, sx + 1] == 0:
            cv2.floodFill(rgb, mask, (sx, sy), (255, 255, 255), (TOL,) * 3, (TOL,) * 3, flags)

    ys, xs = np.where(rgb.min(axis=2) < TRIM_THRESHOLD)
    img = Image.fromarray(rgb[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1])

    scale = LONG_EDGE / max(img.size)
    if scale < 1:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)

    img.save(args.dst, quality=88, method=6)
    print(f"{os.path.basename(args.dst)}  {img.size[0]}x{img.size[1]}  {os.path.getsize(args.dst) // 1024} kB")


if __name__ == "__main__":
    main()
