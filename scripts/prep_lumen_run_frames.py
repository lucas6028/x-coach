"""Build Lumen's run-cycle sprite sheet out of the one still cutout we have of her.

WHY THIS EXISTS. The analysis waiting state (frontend/src/components/LumenLoader.tsx) shows the
mascot running while the pipeline works. There is exactly one drawing of Lumen -- lumen-full.png,
a front-facing standing pose -- and no sprite sheet, so the first cut of that loader animated the
single image with CSS transforms. Moving one picture up and down is not a run cycle: with the feet
welded together and the arms frozen mid-wave, the only thing a bounce can read as is a hop, and
with a lean added, a hop on a board.

So the frames are BUILT here. The cutout is separated into the parts that move independently on a
running figure -- flame, two arms, two shoes, and the body they hang off -- and each output frame
re-poses those parts. The result is genuine frame-by-frame animation (eight different drawings),
not one drawing under a transform, and it stays exactly on model because every pixel still comes
from the original art.

WHAT THE SEPARATION RELIES ON. Nothing here is a general "segment a character" routine; it is four
observations about this one 440x440 file, each verified against the pixels before being written
down:

  * THE FLAME is the warm-coloured pixels above the headband (y < 205) that lie OUTSIDE the ball.
    The row cut alone also takes the warm shading off the top of the head, and the hole that leaves
    opens up the moment the flame sways. Cutting along the ball's own arc instead means the flame's
    boundary IS a circle centred on the ball -- so rotating it about that centre slides the base
    along the head's rim and can never open a gap, whatever the angle.
  * THE ARMS are the DARK pixels outside the ball's silhouette between y = 225 and y = 325. The
    ball spans x = 118..322 through that band, so "dark and outside it" is the arm and nothing
    else. Colour alone does not work: the whistle cord is the same brown family.
  * THE SHOES are everything opaque below y = 366 except the strip of body that shows between them
    (x = 205..235) and the painted shadow they stand on. Matching them by their navy would drop the
    white soles and laces, which are nearer to the body's cream than to the shoe's blue; matching
    the shadow by row alone would eat the bottom third of both shoes, since the two overlap. What
    separates them is warmth: the shadow runs 60+ redder than it is blue, the shoes (navy uppers,
    near-neutral soles) never get past 32.
  * THE BALL IS A CIRCLE, centre (220, 270), radius 108.5 -- fitted from its measured half-widths
    at y = 310/340/360, which agree to within a pixel. That is what makes moving a shoe possible:
    the body behind it can be reconstructed instead of leaving a hole.

The painted-on ground shadow (the tan ellipse under the feet, y >= 378) is DROPPED. It belongs to
a standing pose: left in, it lifts and tilts with the body, which is precisely how a skateboard
behaves. The loader draws its own contact shadow on the floor instead.

Run it from the repository root:

    .venv\\Scripts\\python.exe scripts/prep_lumen_run_frames.py

It writes frontend/public/lumen/lumen-run.webp -- eight frames laid out left to right, each square
and 2x the loader's on-screen size. Pass --debug to also get the frames as separate PNGs and a
contact sheet, which is the only practical way to judge a cycle without a browser.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPO_ROOT / "frontend" / "public" / "lumen" / "lumen-full.png"
SHEET = REPO_ROOT / "frontend" / "public" / "lumen" / "lumen-run.webp"

# --- the four observations above, as numbers -------------------------------------------------
FLAME_BELOW = 205  # last row of flame; the headband starts two rows later
ARM_BAND = (232, 312)  # rows where an arm can be found
SPECK = 60  # px; anything smaller than this in a part is antialiasing, not a limb
BALL_EDGE = (118, 322)  # the ball's left/right silhouette through that band
SHOE_ABOVE = 366  # first row of shoe
SHOE_GAP = (205, 235)  # the body visible between the two shoes, which is NOT shoe
SHADOW_ABOVE = 378  # first row of the painted-on ground shadow
SHADOW_WARMTH = 60  # how much redder than blue that shadow is, and the shoes over it are not
BALL = (220.0, 270.0, 108.5)  # centre x, centre y, radius
REBUILD_FROM = 362  # the row the ball's rebuilt bottom starts from, just clear of the shoes

# Pivots. Each is where the part actually hinges on a running body -- and the arm and flame ones sit
# INSIDE the body on purpose. Hinging an arm at the point where it leaves the silhouette swings the
# hand away from a body that never follows it, opening a gap; hinging it at the shoulder the ball
# would have if it had one sweeps the join through the body's own interior, and since arms and
# flame are drawn before the body, the ball covers every part of that arc that should not be seen.
# The flame's is the ball's exact centre, for the reason given above: its cut edge is that circle,
# so a rotation about the centre slides it along the rim instead of lifting it off.
# A shoe, which is never overlapped, rolls over the ball of the foot rather than its middle.
PIVOT_ARM_L = (150.0, 272.0)
PIVOT_ARM_R = (290.0, 272.0)
PIVOT_FLAME = (BALL_CENTRE := (220.0, 270.0))
PIVOT_SHOE_L = (196.0, 392.0)
PIVOT_SHOE_R = (246.0, 392.0)

FRAMES = 8
OUT_SIZE = 344  # 2x the loader's 172px stage, so the sheet is sharp on a retina display
PAD = 44  # working room around the source, so a lifted, stretched frame cannot clip the flame

# --- how far each part travels ----------------------------------------------------------------
# Deliberately large. This is a 172px mascot on screen: a stride that would look correct on a
# life-size figure disappears entirely at that size, and the whole point of the exercise is that
# the run is legible at a glance.
STRIDE = 26.0  # px a foot travels forward/back over one stride
LIFT = 34.0  # px a foot rises at the top of its swing
TOE_OFF = 26.0  # degrees a shoe rotates through its swing
BOB = 15.0  # px the body rises between footfalls
SQUASH = 0.05  # fraction of squash/stretch on the body at contact / at float
ARM_SWING = 16.0  # degrees each arm swings, counter to the leg on its own side
FLAME_LEAN = 7.0  # degrees the flame trails
FLAME_COLLAR = 14  # px of flame the head keeps, so its base never leaves a bare arc


def load_source() -> np.ndarray:
    if not SOURCE.exists():
        raise SystemExit(f"missing source art: {SOURCE}")
    return np.array(Image.open(SOURCE).convert("RGBA")).astype(np.int16)


def split_parts(art: np.ndarray) -> dict[str, np.ndarray]:
    """Cut the cutout into the parts that move independently, plus the body they hang off."""
    r, g, b, alpha = art[..., 0], art[..., 1], art[..., 2], art[..., 3]
    height, width = alpha.shape
    ys, xs = np.mgrid[0:height, 0:width]
    opaque = alpha > 60
    mean = (r + g + b) / 3

    shadow = opaque & (ys >= SHADOW_ABOVE) & ((r - b) > SHADOW_WARMTH)
    warm = opaque & (r > 200) & (g > 110) & (b < 150)

    ball_cx, ball_cy, ball_r = BALL
    from_centre = (xs - ball_cx) ** 2 + (ys - ball_cy) ** 2
    flame = warm & (ys < FLAME_BELOW) & (from_centre > ball_r**2)
    # The body keeps a collar of the flame's innermost pixels. The fitted circle runs a little
    # inside the drawn head at the crown, so a clean cut hands those pixels to the flame and they
    # leave a bare arc behind the moment it sways. Letting both layers own the collar -- the body
    # is drawn last, so its copy stays put -- covers the join instead of chasing the fit.
    collar = flame & (from_centre <= (ball_r + FLAME_COLLAR) ** 2)
    arms = (
        opaque
        & (mean < 165)
        & (ys >= ARM_BAND[0])
        & (ys <= ARM_BAND[1])
        & ((xs < BALL_EDGE[0]) | (xs > BALL_EDGE[1]))
    )
    shoes = opaque & (ys >= SHOE_ABOVE) & ~shadow & ~((xs >= SHOE_GAP[0]) & (xs <= SHOE_GAP[1]))

    def layer(mask: np.ndarray) -> np.ndarray:
        # Colour thresholds leave crumbs behind -- a few antialiased pixels off the edge of a shoe
        # or a hand. On a still they are invisible; once the part they were cut from moves away and
        # they do not, they read as dirt on the screen. Anything too small to be a limb is dropped.
        labels, count = ndimage.label(mask)
        if count:
            keep = np.zeros(count + 1, dtype=bool)
            sizes = ndimage.sum(mask, labels, range(1, count + 1))
            keep[1:] = sizes >= SPECK
            mask = keep[labels]
        out = art.copy()
        out[~mask] = 0
        return out

    body_mask = opaque & ~(flame & ~collar) & ~arms & ~shoes & ~shadow
    return {
        "flame": layer(flame),
        "arm_l": layer(arms & (xs < width // 2)),
        "arm_r": layer(arms & (xs >= width // 2)),
        "shoe_l": layer(shoes & (xs < SHOE_GAP[0])),
        "shoe_r": layer(shoes & (xs > SHOE_GAP[1])),
        "body": rebuild_ball_bottom(layer(body_mask)),
    }


def rebuild_ball_bottom(body: np.ndarray) -> np.ndarray:
    """Give the body back the round bottom the shoes were drawn over.

    Lifting a shoe exposes the part of the ball it stood in front of, and a flat cutout has nothing
    behind it. Two things happen here, and both are needed:

      * everything below REBUILD_FROM is CUT first -- otherwise the strip of body that showed
        between the two shoes survives as a rectangular tab hanging off the bottom;
      * everything inside the circle that the shoes took is FILLED, each column carrying down the
        colour it has at REBUILD_FROM (a row safely above the shoes, so the sample is ball and not
        the dark edge of a shoe), darkened toward the silhouette the way the ball is shaded.

    Most of this is covered by a shoe in most frames. It is the few pixels of sliver either side of
    a lifted foot that decide whether the figure has a bottom or a notch.
    """
    cx, cy, radius = BALL
    height, width = body.shape[:2]
    out = body.copy()
    sample = out.copy()
    out[REBUILD_FROM:, :] = 0
    for x in range(width):
        dx = x - cx
        if abs(dx) >= radius:
            continue
        edge = int(cy + math.sqrt(radius * radius - dx * dx))
        column = np.where(sample[:REBUILD_FROM, x, 3] > 60)[0]
        if column.size == 0:
            continue
        source = sample[int(column.max()), x].astype(float)
        span = max(1, edge - REBUILD_FROM)
        for y in range(REBUILD_FROM, min(edge + 1, height)):
            if out[y, x, 3] > 60:
                continue
            shade = 1.0 - 0.18 * ((y - REBUILD_FROM) / span)
            out[y, x, :3] = np.clip(source[:3] * shade, 0, 255)
            out[y, x, 3] = source[3]
    return out


def pad(layer: np.ndarray) -> np.ndarray:
    """Put a part on the taller working canvas.

    The source is cropped tight to the standing pose, so a stretched, lifted frame runs the flame
    straight off the top edge. Every part moves onto a padded canvas together and the pivots move
    with them, so nothing in the posing maths has to know about it.
    """
    height, width = layer.shape[:2]
    out = np.zeros((height + 2 * PAD, width + 2 * PAD, 4), dtype=layer.dtype)
    out[PAD : PAD + height, PAD : PAD + width] = layer
    return out


def shift(pivot: tuple[float, float]) -> tuple[float, float]:
    return (pivot[0] + PAD, pivot[1] + PAD)


def place(
    layer: np.ndarray,
    pivot: tuple[float, float],
    *,
    angle: float = 0.0,
    scale: tuple[float, float] = (1.0, 1.0),
    move: tuple[float, float] = (0.0, 0.0),
) -> Image.Image:
    """Rotate/scale a part about its pivot and move it, on the source's own canvas.

    PIL's transform takes the map from OUTPUT back to INPUT, so the matrix below is the inverse of
    "scale about the pivot, then rotate about it, then translate" -- the order the caller thinks in.
    """
    image = Image.fromarray(layer.astype("uint8"), "RGBA")
    theta = math.radians(angle)
    cos, sin = math.cos(theta), math.sin(theta)
    sx, sy = scale
    a, b = cos / sx, sin / sx
    d, e = -sin / sy, cos / sy
    px, py = pivot
    ox, oy = px + move[0], py + move[1]
    c = px - a * ox - b * oy
    f = py - d * ox - e * oy
    return image.transform(image.size, Image.AFFINE, (a, b, c, d, e, f), resample=Image.BICUBIC)


def foot(phase: float) -> tuple[float, float, float]:
    """Where one foot is at `phase` of its own stride: (dx, dy, degrees).

    First half is stance -- planted, travelling backwards under the body. Second half is swing --
    off the ground, arcing forward, toe rotating up and then down into the next contact. Contact is
    at phase 0, which is also where the body is at its lowest.
    """
    if phase < 0.5:
        t = phase / 0.5
        return STRIDE * (1.0 - 2.0 * t), 0.0, -4.0 * t
    t = (phase - 0.5) / 0.5
    return (
        STRIDE * (2.0 * t - 1.0),
        -LIFT * math.sin(math.pi * t),
        -TOE_OFF * math.sin(math.pi * t) - 4.0 + 4.0 * t,
    )


def compose(parts: dict[str, np.ndarray], phase: float, size: int) -> Image.Image:
    """One frame of the cycle."""
    left_dx, left_dy, left_rot = foot(phase)
    right_dx, right_dy, right_rot = foot((phase + 0.5) % 1.0)

    # Two footfalls per cycle, so the body rises and falls twice: lowest at each contact, highest
    # between them. Contact also squashes it and float stretches it -- the impact frames are what
    # say a foot hit the ground instead of the figure hovering.
    rise = abs(math.sin(2 * math.pi * phase))
    body_dy = -BOB * rise
    squash = (1.0 + SQUASH * (1 - 2 * rise), 1.0 - SQUASH * (1 - 2 * rise))
    # A planted foot cannot ride the full bob or it would leave the floor; letting it take a third
    # of it reads as the leg compressing.
    ground = body_dy * 0.35

    swing = math.cos(2 * math.pi * phase)
    canvas = Image.new("RGBA", (parts["body"].shape[1], parts["body"].shape[0]), (0, 0, 0, 0))

    def stack(image: Image.Image) -> None:
        canvas.alpha_composite(image)

    stack(
        place(
            parts["flame"],
            shift(PIVOT_FLAME),
            angle=FLAME_LEAN * swing,
            scale=(1.0, 1.0 + 0.06 * rise),
            move=(0.0, body_dy),
        )
    )
    # Arms counter the legs: the left arm goes back as the left leg comes forward.
    stack(place(parts["arm_l"], shift(PIVOT_ARM_L), angle=-ARM_SWING * swing, move=(0.0, body_dy)))
    stack(place(parts["arm_r"], shift(PIVOT_ARM_R), angle=ARM_SWING * swing, move=(0.0, body_dy)))
    stack(place(parts["body"], shift((BALL[0], BALL[1] + BALL[2])), scale=squash, move=(0.0, body_dy)))
    stack(
        place(
            parts["shoe_l"],
            shift(PIVOT_SHOE_L),
            angle=left_rot,
            move=(left_dx, ground + left_dy),
        )
    )
    stack(
        place(
            parts["shoe_r"],
            shift(PIVOT_SHOE_R),
            angle=right_rot,
            move=(right_dx, ground + right_dy),
        )
    )
    return canvas.resize((size, size), Image.LANCZOS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=FRAMES)
    parser.add_argument("--size", type=int, default=OUT_SIZE)
    parser.add_argument("--debug", action="store_true", help="also write the frames individually")
    args = parser.parse_args()

    parts = {name: pad(layer) for name, layer in split_parts(load_source()).items()}
    frames = [compose(parts, i / args.frames, args.size) for i in range(args.frames)]

    sheet = Image.new("RGBA", (args.size * args.frames, args.size), (0, 0, 0, 0))
    for i, frame in enumerate(frames):
        sheet.paste(frame, (i * args.size, 0))
    SHEET.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(SHEET, "WEBP", quality=92, method=6)
    print(f"wrote {SHEET.relative_to(REPO_ROOT)}  {sheet.size[0]}x{sheet.size[1]}")

    if args.debug:
        out = REPO_ROOT / ".tmp" / "lumen-run"
        out.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(frames):
            frame.save(out / f"frame{i}.png")
        # Contact sheet on the loader's own navy, which is the only background these are ever seen
        # against -- judging a cycle on white flatters it.
        strip = Image.new("RGBA", (args.size * args.frames, args.size), (32, 34, 63, 255))
        for i, frame in enumerate(frames):
            strip.alpha_composite(frame, (i * args.size, 0))
        strip.save(out / "contact-sheet.png")
        print(f"wrote {out.relative_to(REPO_ROOT)}/contact-sheet.png")


if __name__ == "__main__":
    main()
