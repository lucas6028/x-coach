"""Prepare the movement detail page's artwork: the step figures and the muscle-worked plates.

Sibling of prep_movement_art.py (which does the same job for the library cards) and here for the
same reason: the WebPs it produces are hand-supplied illustrations with specific surgery applied,
and committing the binaries without the recipe would leave nobody able to redo it for the next
movement.

The workflow it supports: drop the source PNGs into the two directories below, run this, commit
the WebPs. The PNGs stay where they were dropped and are gitignored -- they are the source, ~1 MB
each, and the page must not ship them.

    frontend/public/movements/steps/<movement>-<n>.png      one figure per step, in order
    frontend/public/movements/muscles-worked/<movement>.png  anterior + posterior in one image

Three things the conversion does:

  * TRIM. The sources sit in a wide margin of near-white; cropping to the ink is most of the
    saving (~900 KB -> ~20 KB for a step figure).
  * KNOCK OUT. The background is removed to transparency by a flood fill from the four corners,
    so only near-white CONNECTED TO THE BORDER goes. The white inside the figure -- shoes, the
    unhighlighted muscles on the plates -- survives. This matters because the step figures are
    shown on a white card in the overview strip but on a tinted stage in the "How to perform"
    tab, where an opaque plate would render as a visible rectangle.
  * MATCH THE SCALE ACROSS A STEP SET. A movement's step figures are measured and scaled
    TOGETHER, never one at a time -- see prepare_steps for why that is the whole ballgame.

Usage (from the repo root):
    .venv\\Scripts\\python.exe scripts/prep_movement_detail_art.py
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLIC = REPO_ROOT / "frontend" / "public" / "movements"
STEPS_DIR = PUBLIC / "steps"
PLATES_DIR = PUBLIC / "muscles-worked"

# Anything darker than this counts as ink rather than background when finding the figure's box.
WHITE_CUTOFF = 246
# Breathing room kept around the art, in source pixels.
PAD = 12
# Colour distance from the corner pixel that the flood fill still treats as background. Low enough
# that a figure's outline stops it, high enough to take the compression noise in the margin.
FLOOD_THRESHOLD = 26
# The sentinel the fill paints; anything left this colour becomes transparent. Nothing in the
# source art is near it.
SENTINEL = (255, 0, 255)

# Rendered sizes, x3 for a high-DPI screen: a step figure is ~132 px tall in the strip, a plate
# ~440 px wide in its card.
STEP_HEIGHT = 420
PLATE_WIDTH = 1000


def _flatten(path: Path) -> Image.Image:
    """The source as opaque RGB. Composited onto white FIRST because a source carrying an alpha
    channel would otherwise defeat both the bounding-box search and the flood fill."""
    image = Image.open(path).convert("RGBA")
    flat = Image.new("RGB", image.size, (255, 255, 255))
    flat.paste(image, mask=image.split()[3])
    return flat


def _ink_box(flat: Image.Image, path: Path) -> tuple[int, int, int, int]:
    """The box around everything that is not background, padded."""
    mask = flat.convert("L").point(lambda p: 255 if p < WHITE_CUTOFF else 0)
    box = mask.getbbox()
    if box is None:
        raise SystemExit(f"{path.name} is blank -- nothing to crop")
    return (
        max(0, box[0] - PAD),
        max(0, box[1] - PAD),
        min(flat.width, box[2] + PAD),
        min(flat.height, box[3] + PAD),
    )


def _trim_and_knockout(path: Path) -> Image.Image:
    flat = _flatten(path)
    out = flat.crop(_ink_box(flat, path))

    fill = out.copy()
    corners = (
        (0, 0),
        (fill.width - 1, 0),
        (0, fill.height - 1),
        (fill.width - 1, fill.height - 1),
    )
    for corner in corners:
        ImageDraw.floodfill(fill, corner, SENTINEL, thresh=FLOOD_THRESHOLD)
    alpha = Image.new("L", fill.size, 255)
    alpha.putdata([0 if px == SENTINEL else 255 for px in fill.get_flattened_data()])

    out = out.convert("RGBA")
    out.putalpha(alpha)
    return out


def _save(image: Image.Image, path: Path) -> None:
    image.save(path, "WEBP", quality=88, method=6)
    print(f"{path.parent.name}/{path.name} {image.size} {path.stat().st_size // 1024} KB")


def prepare_plates(directory: Path) -> int:
    """One plate per movement, each independent — nothing to keep in step with."""
    sources = sorted(directory.glob("*.png"))
    for source in sources:
        art = _trim_and_knockout(source)
        if art.width > PLATE_WIDTH:
            art = art.resize(
                (PLATE_WIDTH, round(art.height * PLATE_WIDTH / art.width)), Image.LANCZOS
            )
        _save(art, source.with_suffix(".webp"))
    return len(sources)


def prepare_steps(directory: Path) -> int:
    """The step figures, ONE MOVEMENT AT A TIME and to a shared scale.

    This is the part that cannot be done per image. Trimming each figure to its own ink and then
    scaling it to a fixed height makes every pose the same height on screen -- so the squat at the
    bottom of the rep, which the artist drew ~30% shorter than the standing pose, renders as a
    LARGER person than the one standing next to it. The figures have to be measured together:

      * one scale for the whole set, taken from the tallest figure in it, so the drawn proportions
        survive and a crouch reads as a crouch;
      * one output canvas for the whole set, with each figure bottom-aligned on it, so the feet
        share a ground line even though the sources place them at different heights;
      * identical output dimensions, so whatever the page does to size them applies equally.

    The set is everything sharing a `<movement>-<n>.png` prefix.
    """
    groups: dict[str, list[Path]] = {}
    for source in sorted(directory.glob("*.png")):
        groups.setdefault(source.stem.rsplit("-", 1)[0], []).append(source)

    made = 0
    for movement, sources in groups.items():
        flats = [_flatten(s) for s in sources]
        boxes = [_ink_box(f, s) for f, s in zip(flats, sources)]

        scale = STEP_HEIGHT / max(box[3] - box[1] for box in boxes)
        canvas = (
            round(max(box[2] - box[0] for box in boxes) * scale),
            STEP_HEIGHT,
        )
        print(f"{movement}: {len(sources)} steps, shared scale {scale:.3f}, canvas {canvas}")

        for source, box in zip(sources, boxes):
            art = _trim_and_knockout(source)
            art = art.resize(
                (max(1, round(art.width * scale)), max(1, round(art.height * scale))),
                Image.LANCZOS,
            )
            # Transparent, so the padding that makes every step the same shape is invisible.
            frame = Image.new("RGBA", canvas, (0, 0, 0, 0))
            frame.paste(art, ((canvas[0] - art.width) // 2, canvas[1] - art.height), art)
            _save(frame, source.with_suffix(".webp"))
            made += 1
    return made


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=Path, default=STEPS_DIR)
    parser.add_argument("--plates", type=Path, default=PLATES_DIR)
    args = parser.parse_args()

    made = prepare_steps(args.steps) + prepare_plates(args.plates)
    if not made:
        raise SystemExit("no source PNGs found -- drop them in first (see the module docstring)")


if __name__ == "__main__":
    main()
