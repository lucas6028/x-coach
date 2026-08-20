"""Prepare the movement-card illustrations in frontend/public/movements/.

Unlike its neighbours in scripts/ this one calls nothing in src/ -- it is an asset pipeline, not a
CLI over a research module. It lives here because the eight PNGs it produces are third-party art
with specific surgery applied, and committing the binaries without the recipe would leave nobody
able to say what was cut, or to redo it against a new source.

The sources are the exercise_library_muse-spark reference's `public/images/`: scraped stock
illustrations, 6.9 MB, in no single style. Before they can ship they need

  * the stock-agency watermark bar cropped off row.png,
  * a third-party wordmark erased from pushup.png,
  * their backgrounds -- flat white, a faint grid, a photograph of a swimming pool -- knocked out
    to transparency, so the eight read as one set on the card's pale tile,
  * a trim and a resize, to ~1 MB for the set.

NOT removed, and worth knowing before reusing these: lunge.png's subject wears Nike-branded kit
(a swoosh on the shorts and on both shoes). Erasing it means retouching over the garment rather
than cropping, so it is left as-is and called out here.

Usage (from the repo root):
    .venv\\Scripts\\python.exe scripts/prep_movement_art.py --src <reference>/public/images
"""

import argparse
import os

import cv2
import numpy as np
from PIL import Image, ImageFilter

# Surgery applied BEFORE the background knockout.
#   crop:  keep only this box -- cuts the watermark strip off row.png
#   erase: paint these boxes the background colour, so the knockout takes them with it
PRE = {
    "row.png": {"crop": (0, 0, 1200, 1496)},
    "pushup.png": {"erase": [(1250, 880, 1600, 1200)]},
}

# Colour distance from the SEED (not from the neighbouring pixel) that still counts as background.
# Progressive matching was tried first and is wrong here: at any tolerance loose enough to cross a
# background gradient it also walks across the figure's own soft shading and dissolves the body,
# leaving only its outline. Fixed-range cannot, because skin and shirt are nowhere near white.
TOL = 16
# pushup/row sit on a faint grid: its lines are ~18 levels off the white around them, so at the
# default tolerance the fill stops at the first line and every cell beyond it stays opaque.
# squat's background is a swimming pool, which needs a wider band still.
TOL_BY_IMAGE = {"pushup.png": 30, "row.png": 30, "squat.png": 44}
# squat only: the pool survives any tolerance the figure also survives, so it is keyed out by hue
# instead. Green AND blue clearly above red is water; it is not skin (r>g>b), not the violet shirt
# (blue high but green low), and not the black or white of the kit.
TEAL_KEY = {"squat.png"}
# row only: the grid cells its figure and the bench enclose are unreachable from the border, so the
# flood never gets to them. Nothing on that figure is white.
WHITE_KEY = {"row.png"}
LONG_EDGE = 560


def knockout(img: Image.Image, tol: int, teal_key: bool, white_key: bool) -> Image.Image:
    """Make the background transparent by flood-filling inwards from the border."""
    rgb = np.asarray(img.convert("RGB"))
    h, w, _ = rgb.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    flags = 4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE
    lo = up = (tol, tol, tol)

    seeds = [(x, 0) for x in range(0, w, 4)] + [(x, h - 1) for x in range(0, w, 4)]
    seeds += [(0, y) for y in range(0, h, 4)] + [(w - 1, y) for y in range(0, h, 4)]

    # Not every border pixel is background. lunge.png's subject is cropped at the top of his own
    # canvas, so his hair runs off the edge -- seeded there, the fill eats the whole head. A colour
    # only counts as background if it holds a real share of the border: the dark top band of
    # squat's pool photo spans the full width and qualifies, 60px of hair does not.
    edge = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    keys, counts = np.unique(edge // 16, axis=0, return_counts=True)
    bg_keys = {tuple(k) for k, c in zip(keys, counts) if c >= 0.03 * len(edge)}

    work = rgb.copy()
    for sx, sy in seeds:
        if mask[sy + 1, sx + 1] == 0 and tuple(rgb[sy, sx] // 16) in bg_keys:
            cv2.floodFill(work, mask, (sx, sy), 0, lo, up, flags)

    bg = mask[1:-1, 1:-1].astype(bool)
    if teal_key:
        r, g, b = (rgb[:, :, i].astype(int) for i in range(3))
        bg |= (g > r + 12) & (b > r + 20)
    if white_key:
        bg |= rgb.min(axis=2) > 232
        # ...except the eyes, the drops of sweat and the wordmark on the shirt, which come out as
        # pinprick holes. Any background island under 1500px is put back.
        n, lab, st, _ = cv2.connectedComponentsWithStats(bg.astype(np.uint8), connectivity=8)
        for i in range(1, n):
            if st[i, cv2.CC_STAT_AREA] < 1500:
                bg[lab == i] = False

    # Keep only the largest remaining blob. Whatever the flood and the keys both miss -- the wooden
    # decking in squat's corners, a stray grid cell -- is a separate island, while each figure is
    # one connected piece, its equipment included, held in its hands.
    n, labels, stats, _ = cv2.connectedComponentsWithStats((~bg).astype(np.uint8), connectivity=8)
    if n > 1:
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        bg |= labels != biggest

    out = img.convert("RGBA")
    alpha = Image.fromarray(np.where(bg, 0, 255).astype(np.uint8), "L")
    # A one-pixel blur softens the cut edge; without it the knockout reads as a sticker.
    out.putalpha(alpha.filter(ImageFilter.GaussianBlur(0.8)))
    return out


def main() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="reference public/images directory")
    ap.add_argument("--dst", default=os.path.join(root, "frontend", "public", "movements"))
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    total = 0
    for name in sorted(os.listdir(args.src)):
        if not name.endswith(".png"):
            continue
        img = Image.open(os.path.join(args.src, name)).convert("RGB")
        pre = PRE.get(name, {})
        if "crop" in pre:
            img = img.crop(pre["crop"])
        for box in pre.get("erase", []):
            img.paste(img.getpixel((2, 2)), box)

        img = knockout(img, TOL_BY_IMAGE.get(name, TOL), name in TEAL_KEY, name in WHITE_KEY)
        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)
        scale = LONG_EDGE / max(img.size)
        if scale < 1:
            img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)

        out = os.path.join(args.dst, name)
        img.save(out, optimize=True)
        kb = os.path.getsize(out) // 1024
        total += kb
        print(f"{name:20s} {img.size[0]}x{img.size[1]}  {kb} kB")
    print(f"total {total} kB")


if __name__ == "__main__":
    main()
