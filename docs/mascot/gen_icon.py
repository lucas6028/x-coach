#!/usr/bin/env python3
"""Cut Lumen out of its cream background and build an app-icon set from the main visual.

Strategy: corner flood-fill marks only background cream that is connected to the
border, so the enclosed near-white body is protected (a plain color-key would eat it).
Then feather the alpha, auto-crop to content, and composite onto branded tiles.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SP = Path(__file__).parent
NIGHT = (32, 34, 63)      # #20223F
CREAM = (253, 247, 231)   # #FDF7E7


def cutout(src: Path, flood_thresh: int = 20, color_dist: float = 46,
           feather: float = 0.8) -> Image.Image:
    """Return an RGBA image with the cream background made transparent.

    Two signals are UNIONed so the near-white body and its shadowed lower face both
    stay opaque:
      1. flood-fill from bg-only seeds (corners + side midpoints, NOT under the body,
         which used to let the fill creep up through the soft floor shadow into the jaw);
      2. colour distance from the measured cream — any pixel clearly unlike cream is
         forced opaque, recovering shadowed jaw/whistle pixels the flood nicked.
    Alpha is then binarised (kills the semi-transparency that read as black on navy).
    """
    im = Image.open(src).convert("RGB")
    w, h = im.size
    arr = np.asarray(im).astype(int)
    border = np.concatenate([arr[:6].reshape(-1, 3), arr[-6:].reshape(-1, 3),
                             arr[:, :6].reshape(-1, 3), arr[:, -6:].reshape(-1, 3)])
    bg_cream = np.median(border, axis=0)
    dist = np.sqrt(((arr - bg_cream) ** 2).sum(-1))

    work = im.copy()
    SENT = (255, 0, 255)
    seeds = [(2, 2), (w - 3, 2), (2, h - 3), (w - 3, h - 3),
             (w // 2, 2), (2, h // 2), (w - 3, h // 2)]  # no bottom-centre seed
    for s in seeds:
        ImageDraw.floodfill(work, s, SENT, thresh=flood_thresh)
    flood_bg = np.all(np.asarray(work) == np.array(SENT), axis=-1)

    keep = (~flood_bg) | (dist > color_dist)
    alpha = np.where(keep, 255, 0).astype(np.uint8)
    a_img = Image.fromarray(alpha, "L")
    # close (dilate->erode) seals the thin gaps the red whistle-cord punches through, so
    # the belly the flood leaked into becomes a fully-enclosed hole...
    a_img = a_img.filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.MinFilter(9))
    # ...then hole-fill forces any transparent region NOT connected to the border opaque:
    # THIS is what kills the big belly hole that read as dark stage colour on the loader
    # (close alone can't span a hole that large). Median then clears residual speckle.
    a_img = _fill_holes(a_img)
    a_img = a_img.filter(ImageFilter.MedianFilter(5))
    if feather:
        a_img = a_img.filter(ImageFilter.GaussianBlur(feather))
    out = im.convert("RGBA")
    out.putalpha(a_img)
    return out


def _fill_holes(a_img: Image.Image) -> Image.Image:
    """Force any transparent region NOT connected to the image border to opaque.

    Flood-fill leaks can carve the near-white body interior (belly/chest) into a big
    transparent hole; on white it's invisible, but on the dark loader stage it shows
    through as black. We binarise, flood-fill the transparent EXTERIOR from a corner,
    and anything transparent the flood never reached is an interior hole -> make opaque.
    """
    a = np.asarray(a_img)
    binm = np.where(a > 128, 255, 0).astype(np.uint8)  # opaque=255, transparent=0
    # .copy() forces an owned buffer; a fromarray image shares numpy's read-only buffer
    # and floodfill's pixel writes would silently no-op.
    inv = Image.fromarray((255 - binm).astype(np.uint8), "L").copy()  # transparent=255
    ImageDraw.floodfill(inv, (0, 0), 128)  # mark border-connected transparent
    still_transparent = np.asarray(inv) == 255         # enclosed holes only
    binm[still_transparent] = 255
    return Image.fromarray(binm, "L")


def content_bbox(rgba: Image.Image, pad_frac: float = 0.06) -> tuple:
    a = np.asarray(rgba.split()[-1])
    ys, xs = np.where(a > 24)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    return x0, y0, x1, y1


def _dark_face_line(rgba: Image.Image, x0, y0, x1, y1) -> int:
    """Find the y just below the eyes/smile using the navy (very dark) face pixels."""
    arr = np.asarray(rgba)
    r, g, b, a = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]
    dark = (a > 40) & (r < 70) & (g < 70) & (b < 95)
    ys, xs = np.where(dark)
    # restrict to the upper-central head region to avoid navy sneakers at the feet
    keep = (ys > y0) & (ys < y0 + (y1 - y0) * 0.75) & \
           (xs > x0 + (x1 - x0) * 0.2) & (xs < x0 + (x1 - x0) * 0.8)
    if not keep.any():
        return y0 + int((y1 - y0) * 0.6)
    return int(ys[keep].max())


def solidify_lower(rgba: Image.Image, start_frac: float = 0.44) -> Image.Image:
    """Fill each column solid from start_frac down to the silhouette's lowest opaque
    pixel. Kills interior semi-transparency in the face/body blob (the shadowed
    chin/whistle notch that read as black on navy) without webbing the upper flame."""
    a = np.asarray(rgba.split()[-1]).copy()
    h, w = a.shape
    y_start = int(h * start_frac)
    solid = a > 128
    for x in range(w):
        rows = np.where(solid[:, x])[0]
        rows = rows[rows >= y_start]
        if rows.size:
            # fill only between this column's own top & bottom opaque -> no box-out
            a[rows.min():rows.max() + 1, x] = 255
    out = rgba.copy()
    out.putalpha(Image.fromarray(a, "L"))
    return out


def square_canvas(rgba: Image.Image, bbox, pad_frac: float, focus: str = "full") -> Image.Image:
    """Crop to bbox (optionally just the head/flame) and center on a transparent square."""
    x0, y0, x1, y1 = bbox
    if focus == "head":
        # bottom = a little chin below the smile (drops the noisy whistle/lower-body
        # zone the matte can't cleanly cut); width = flame/head span only (arms outside)
        face_y = _dark_face_line(rgba, x0, y0, x1, y1)
        y1 = min(y1, face_y + int((y1 - y0) * 0.06))
        a = np.asarray(rgba.split()[-1])
        head_rows = a[y0:y0 + int((y1 - y0) * 0.6), :]
        cols = np.where(head_rows.max(axis=0) > 24)[0]
        if len(cols):
            x0, x1 = int(cols.min()), int(cols.max())
    crop = rgba.crop((x0, y0, x1, y1))
    cw, ch = crop.size
    side = int(max(cw, ch) * (1 + 2 * pad_frac))
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.alpha_composite(crop, ((side - cw) // 2, (side - ch) // 2))
    return canvas


def head_crop_original(src: Path, rgba: Image.Image, bbox, margin=0.09) -> Image.Image:
    """Square crop of flame+face straight from the ORIGINAL image (cream bg intact).

    No matte at all — the character keeps its native cream, which is what the badge/tile
    frame around. Sidesteps every background-removal artifact on the shadowed lower face.
    """
    x0, y0, x1, y1 = bbox
    face_y = _dark_face_line(rgba, x0, y0, x1, y1)
    yb = min(y1, face_y + int((y1 - y0) * margin))
    a = np.asarray(rgba.split()[-1])
    head_rows = a[y0:y0 + int((yb - y0) * 0.6), :]
    cols = np.where(head_rows.max(axis=0) > 24)[0]
    xa, xb = int(cols.min()), int(cols.max())
    px, py = int((xb - xa) * 0.06), int((yb - y0) * 0.05)
    orig = Image.open(src).convert("RGB")
    box = (max(0, xa - px), max(0, y0 - py), min(orig.width, xb + px), min(orig.height, yb + py))
    crop = orig.crop(box)
    cream = tuple(int(v) for v in np.asarray(crop)[:6, :6].reshape(-1, 3).mean(0))
    cw, ch = crop.size
    side = max(cw, ch)
    sq = Image.new("RGB", (side, side), cream)
    sq.paste(crop, ((side - cw) // 2, (side - ch) // 2))
    return sq.convert("RGBA")


def rounded_mask(size: int, radius_frac: float) -> Image.Image:
    m = Image.new("L", (size, size), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, size - 1, size - 1],
                                        radius=int(size * radius_frac), fill=255)
    return m


def compose_badge(crop: Image.Image, size: int, *, rounded=True, radius_frac=0.225,
                  inset=0.80, inner_radius=0.16, ring=True) -> Image.Image:
    """Cream character crop framed as a rounded badge on a navy tile."""
    tile = (rounded_tile(size, radius_frac, NIGHT) if rounded
            else Image.new("RGBA", (size, size), NIGHT + (255,)))
    tile.alpha_composite(glow(size, (255, 178, 58), strength=0.32, cy_frac=0.5, r_frac=0.5))
    isz = int(size * inset)
    off = (size - isz) // 2
    inner = crop.resize((isz, isz), Image.LANCZOS)
    m = rounded_mask(isz, inner_radius)
    # drop shadow under the badge
    sh = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    sh.paste(Image.new("RGBA", (isz, isz), (0, 0, 0, 120)), (off, off + int(size * 0.012)), m)
    tile.alpha_composite(sh.filter(ImageFilter.GaussianBlur(size * 0.02)))
    tile.alpha_composite(Image.composite(inner, Image.new("RGBA", (isz, isz), (0, 0, 0, 0)), m), (off, off))
    if ring:
        ImageDraw.Draw(tile).rounded_rectangle(
            [off, off, off + isz - 1, off + isz - 1], radius=int(isz * inner_radius),
            outline=(255, 205, 110, 220), width=max(1, size // 170))
    return tile


def rounded_tile(size: int, radius_frac: float, bg) -> Image.Image:
    tile = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    mask = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(mask)
    r = int(size * radius_frac)
    d.rounded_rectangle([0, 0, size - 1, size - 1], radius=r, fill=255)
    fill = Image.new("RGBA", (size, size), bg + (255,))
    tile.paste(fill, (0, 0), mask)
    return tile


def glow(size: int, color, strength: float = 0.5, cy_frac=0.5, r_frac=0.5) -> Image.Image:
    g = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(g)
    cx, cy, r = size // 2, int(size * cy_frac), int(size * r_frac)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color + (int(255 * strength),))
    return g.filter(ImageFilter.GaussianBlur(size * 0.13))


def silhouette(char: Image.Image, color, opacity: float) -> Image.Image:
    """A flat color version of the char shaped by its alpha (for halo/rim backing)."""
    a = char.split()[-1].point(lambda v: int(v * opacity))
    solid = Image.new("RGBA", char.size, color + (255,))
    empty = Image.new("RGBA", char.size, (0, 0, 0, 0))
    return Image.composite(solid, empty, a)


def backing(size, char, pos, color, blur, opacity) -> Image.Image:
    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    layer.alpha_composite(silhouette(char, color, opacity), pos)
    return layer.filter(ImageFilter.GaussianBlur(blur))


def compose_icon(char: Image.Image, size: int, bg, *, rounded=True,
                 radius_frac=0.225, char_frac=0.80, add_glow=None,
                 y_shift=0.0, halo=False) -> Image.Image:
    base = (rounded_tile(size, radius_frac, bg) if rounded
            else Image.new("RGBA", (size, size), bg + (255,)))
    c = char.copy()
    target = int(size * char_frac)
    c.thumbnail((target, target), Image.LANCZOS)
    x = (size - c.width) // 2
    y = (size - c.height) // 2 + int(size * y_shift)
    if add_glow:
        base.alpha_composite(glow(size, add_glow, strength=0.55, cy_frac=0.5, r_frac=0.52))
    if halo:
        # warm outer halo lifts the whole silhouette off the dark ground...
        base.alpha_composite(backing(size, c, (x, y), (255, 202, 112), size * 0.045, 0.9))
        # ...and a tight near-white rim guarantees the shadowed lower face separates
        base.alpha_composite(backing(size, c, (x, y), (255, 248, 233), size * 0.012, 0.95))
    base.alpha_composite(c, (x, y))
    return base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SP / "set-C-genki.png"))
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args()

    rgba = cutout(Path(args.src))
    bbox = content_bbox(rgba)
    print("cutout bbox:", bbox, "of", rgba.size)

    if args.build:
        out = SP / "icons"
        out.mkdir(exist_ok=True)
        # transparent cutouts (reused by the loader animation)
        square_canvas(rgba, bbox, 0.06, focus="head").save(out / "lumen-head.png")
        square_canvas(rgba, bbox, 0.08, focus="full").save(out / "lumen-full.png")
        rgba.save(out / "lumen-cutout.png")

        # ICONS: framed from the ORIGINAL cream crop (no matte -> no dark-face artifacts)
        sq = head_crop_original(Path(args.src), rgba, bbox, margin=0.09)
        sq512 = sq.resize((512, 512), Image.LANCZOS)
        cream_master = rounded_tile(512, 0.225, CREAM)
        cream_master.alpha_composite(Image.composite(
            sq512, Image.new("RGBA", (512, 512), (0, 0, 0, 0)), rounded_mask(512, 0.225)))
        masters = {
            "navy": compose_badge(sq, 512, rounded=True, inset=0.80),
            "cream": cream_master,
            "maskable": compose_badge(sq, 512, rounded=False, inset=0.64, radius_frac=0),
        }
        sizes = [512, 192, 180, 120, 32]
        for name, master in masters.items():
            for s in sizes:
                icon = master.resize((s, s), Image.LANCZOS)
                (icon if name != "cream" else icon).save(out / f"lumen-icon-{name}-{s}.png")
        # favicon.ico (multi-res) from the navy tile
        fav = masters["navy"]
        fav.resize((64, 64), Image.LANCZOS).save(
            out / "favicon.ico", sizes=[(16, 16), (32, 32), (48, 48), (64, 64)])
        print("built icon set in", out)
        return

    if args.preview:
        # character on a checkerboard + on navy, to judge matte quality
        head = square_canvas(rgba, bbox, 0.06, focus="head")
        chk = Image.new("RGBA", head.size, (255, 255, 255, 255))
        d = ImageDraw.Draw(chk)
        step = head.size[0] // 16
        for i in range(0, head.size[0], step):
            for j in range(0, head.size[1], step):
                if (i // step + j // step) % 2:
                    d.rectangle([i, j, i + step, j + step], fill=(205, 205, 215, 255))
        chk.alpha_composite(head)
        chk.convert("RGB").save(SP / "icon-preview-checker.png")
        compose_icon(head, 512, NIGHT, add_glow=(255, 180, 60), char_frac=0.86)\
            .convert("RGB").save(SP / "icon-preview-navy.png")
        print("wrote previews")


if __name__ == "__main__":
    main()
