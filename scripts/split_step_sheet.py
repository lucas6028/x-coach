"""Split a numbered "how to perform" sheet into the one-figure-per-step PNGs the detail page wants.

The upstream step for prep_movement_detail_art.py, and here for the same reason that one exists:
the PNGs under frontend/public/movements/steps/ are gitignored sources, so without the recipe
nobody could redo the cut against a new sheet.

Some step art arrives as five separate figures; some arrives as ONE sheet with all five panels,
their captions, and the arrows and guide lines that say what the pose is doing. What has to come
out is the captions; what has to STAY is the figure and its annotations -- an arrow through the
hips is half of what step 3 means, and the page prints the caption itself beside the figure.

Two rules do the separating:

  * THE FIGURE is one of the N largest connected ink components. A drawn person is a single
    connected mass an order of magnitude bigger than any caption glyph or arrowhead.

  * ITS ANNOTATIONS are the accent-coloured components drawn over that figure's column. Colour is
    what tells an arrow from a caption -- the arrows, dashes and angle labels are violet (blue well
    ahead of red and green), the captions are neutral grey. Position is what tells them from the
    numbered badges and the caret separators, which are the same violet but sit in the caption band
    or between two panels.

Everything else on the sheet is dropped pixel by pixel rather than cropped around, which is the
only way to lose a caption that overlaps a figure's bounding box.

  * UPSCALE THE SET AS A SET. Sheet panels are small -- a figure on a 1893x831 sheet is ~160 px
    tall, against the ~420 px the page wants at 3x. One shared factor, taken from the tallest panel
    (figure plus annotations, which is what the downstream trim will measure), so the drawn
    proportions survive: same rule, same reason as prepare_steps. Sizing it this way leaves the
    downstream resize a no-op, so the art is resampled once, here, where the unsharp pass can put
    back the edge the enlargement costs.

    Note what this buys: prep_movement_detail_art.py scales a step set by its TALLEST panel and
    uses ONE scale for all five, so a step whose arrow reaches higher than the rest makes the whole
    set slightly smaller rather than making that one figure smaller than its neighbours. Keeping
    the annotations costs a little height; it does not distort the set.

Reading order is row-major: figures are grouped into rows by vertical overlap, then sorted left to
right within each row, so `--movement shoulder-bridge` writes shoulder-bridge-1..5.png in the
order the sheet numbers them. Check the printed table against the sheet before trusting it.

Needs Pillow and scipy, both already in .venv and neither in requirements.txt -- the same footing
as prep_movement_detail_art.py's Pillow. This runs offline against a file on disk; nothing it needs
belongs in the web image.

Usage (from the repo root), then run prep_movement_detail_art.py to make the WebPs:

    .venv\\Scripts\\python.exe scripts/split_step_sheet.py \\
        frontend/public/movements/steps/shoulder-bridge-steps.png --movement shoulder-bridge

Thirteen sheets have gone through it -- the whole catalog -- all 5-panel. Where a sheet carries
captions they match the
movement's `steps` text in frontend/src/lib/movementDetail.ts verbatim, which is the cheapest check
that a sheet is the art for the movement you think it is. The sheets live beside their output (and
are gitignored with it); prep_movement_detail_art.py skips them by name, since a whole sheet
joining a step group would silently become the tallest thing the set is scaled by.

    shoulder-bridge  1893x831, 3 figures then 2   arrows on 2/3/5, dashed guide on 4
    push-up          2161x728, one row, captions BELOW the figures   dashed guide on 2,
                     a 45° arc and label on 3, two arrows on 5
    overhead-press   2172x724, one row            no annotations
    row              2172x724, one row            no annotations
    bicep-curl       2172x724, one row            no annotations
    band-pull-apart  2172x724, one row            no annotations
    high-knee        2172x724, one row            no annotations
    sit-up           2149x732, one row, NO captions at all   no annotations
    jumping-jacks    2171x724, one row, NO captions at all   no annotations
    arm-abduction    2164x727, one row            no annotations
    leg-abduction    2172x724, one row            no annotations
    torso-twist      1254x1254, one row on a square canvas   no annotations; the sheet whose
                     panels overlap, see the chevron note by SEPARATOR_AREA_TOLERANCE
    arm-vw           1754x897, one row, and the only sheet with a TITLE ("Arm V to W (Standing)")
                     no annotations -- the title is violet, so only position drops it: its
                     glyphs sit 56 px above the first figure, past ANNOTATION_REACH_Y

Eleven of the thirteen are the useful negative case: their violet chevron separators and numbered
badges are exactly the accent colour an arrow would be, and every one comes out with zero
annotations kept. The badges fail the horizontal test (they sit left of their figure's column, in
the caption band); the chevrons fail it on eleven sheets by sitting outside every figure's box, and
on Torso Twist -- where the boxes overlap and two of them do NOT fail it -- they are dropped as a
row instead. The two caption-less sheets are the cleanest of all, at 56 and 37 ink components for
the whole sheet against 361-739 for a captioned one.

Only two of the thirteen carry annotations at all. That is worth knowing before hunting a bug: a
run reporting "0 annotation(s)" on a new sheet is the norm, not a failure.

The chevron count in the header line is the check that matters instead. On a one-row 5-panel sheet
it should read 4 -- one per gap. On a two-row sheet it should read 0, because the filter needs all
of them in one horizontal band and cannot see a row-split set (harmless: position already excludes
them there). Any other number means the congruence grouping caught something it should not have, or
missed one; look at the panels before trusting the output.

Worth knowing about the two sheets whose figures carry props: band-pull-apart's red band and
bicep-curl's dumbbells are drawn TOUCHING the hands, so they are part of the figure's own connected
component and survive. A prop drawn detached would not -- the accent test is tuned for violet and
would drop a red band as firmly as it drops a caption. Check the printed panel widths against the
sheet when a new prop appears: a band that vanished would show up as a suddenly narrow panel.

    band-pull-apart  panel 1 x99-342 vs panel 5 x1806-2152 -- the wide ones are the band, kept
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[1]
STEPS_DIR = REPO_ROOT / "frontend" / "public" / "movements" / "steps"

# Anything darker counts as ink rather than background. Same cutoff as prep_movement_detail_art,
# and for the same reason: the sheets are near-white (254), not white.
WHITE_CUTOFF = 246
# How far ahead of red and green a component's mean blue has to be to count as annotation rather
# than caption. Measured on the shoulder-bridge sheet: arrows and dashes score 45-68, caption
# glyphs 0-1, the pale ground shadows 5-11.
ACCENT_MARGIN = 20
# How far outside the figure's box an annotation may sit and still be taken as its own -- and the
# two axes are deliberately different, because the sheets' layout is. An annotation belongs to the
# figure it is drawn OVER, so it stays inside that figure's column (slack only for a dash that
# overshoots an edge) but may sit well above or below it: an arrow overhangs by ~15 px and the
# "45°" label hangs 36 px under the elbow. One reach loose enough for that label would also swallow
# the violet caret separators, which sit 31 px to the SIDE. Two reaches separate them cleanly, and
# the numbered badges fail both tests -- they are 74-226 px into the caption band.
ANNOTATION_SLACK_X = 12
ANNOTATION_REACH_Y = 45
# Ink this small is a stray speck, not a mark worth carrying.
MIN_MARK = 12
# The chevrons a sheet draws BETWEEN panels are the same violet as an arrow, and on twelve of the
# thirteen sheets position alone excludes them: they sit in the gap, outside every figure's column.
# Torso Twist is the exception. Its figures are seated with the legs extended, which makes the
# panels' bounding boxes OVERLAP, so two of its four chevrons land inside a neighbour's box and
# would be pasted onto it as if they were annotations.
#
# Two simpler fixes were measured against the two annotated sheets and both fail:
#
#   * DISTANCE to the figure's ink does not separate them. Genuine annotations run 5-31 px from
#     the ink they refer to; the two stray chevrons are 28 px and 43 px. The ranges overlap.
#   * REQUIRING the mark to sit over the figure's ink in its own rows throws away nearly every
#     real annotation, because a dashed guide runs ALONGSIDE the body rather than across it.
#
# What is left is what the chevrons actually are: a row of them, one per gap, all drawn identically
# -- which no annotation is. Tolerances for "identically": relative area, and the top edge and
# height in pixels.
SEPARATOR_AREA_TOLERANCE = 0.12
SEPARATOR_BAND = 8
SEPARATOR_HEIGHT = 5
# Margin kept around each panel. Only has to be non-zero -- the downstream trim re-finds the ink.
MARGIN = 16
# The ink height each set is enlarged to. prep_movement_detail_art fits a set into STEP_HEIGHT=420
# including its PAD=12 on each side, so a panel this tall leaves its resize at ~1.0 and the art is
# resampled once rather than twice.
INK_HEIGHT = 396
# Enough to put back the edge a ~2x enlargement softens, gentle enough not to ring. The alpha
# composite afterwards keeps any halo off the background, which the flood-fill knockout downstream
# would otherwise read as ink.
UNSHARP = ImageFilter.UnsharpMask(radius=2, percent=70, threshold=3)


def _over(mark, figure) -> bool:
    """Is `mark`'s box in `figure`'s column -- tight across, generous above and below?"""
    my, mx = mark
    fy, fx = figure
    return (
        mx.start >= fx.start - ANNOTATION_SLACK_X
        and mx.stop <= fx.stop + ANNOTATION_SLACK_X
        and my.start >= fy.start - ANNOTATION_REACH_Y
        and my.stop <= fy.stop + ANNOTATION_REACH_Y
    )


def _separator_row(accents, sizes, boxes, ordered) -> set[int]:
    """The labels of the chevrons drawn between panels, or an empty set if none can be identified.

    A group qualifies only if it has exactly one member per gap AND those members interleave the
    figures in reading order -- second chevron between the second and third figure, and so on.
    That is a strong enough shape that nothing else on any of the thirteen sheets forms one: not
    the numbered badges (there are `count` of those, not `count - 1`), not a dashed guide (its
    dashes lie on a diagonal, so they never share one horizontal band).

    It only fires on a single-row sheet, since it needs every chevron in one band and a two-row
    sheet splits them across two. That is not a gap: a two-row sheet has room between its panels,
    so `_over` already excludes them there.
    """
    centres = [(boxes[f - 1][1].start + boxes[f - 1][1].stop) / 2 for f in ordered]
    for anchor in accents:
        top, height = boxes[anchor - 1][0].start, _height(boxes[anchor - 1])
        group = sorted(
            (
                mark
                for mark in accents
                if abs(int(sizes[mark]) - int(sizes[anchor])) / max(sizes[mark], sizes[anchor])
                < SEPARATOR_AREA_TOLERANCE
                and abs(boxes[mark - 1][0].start - top) < SEPARATOR_BAND
                and abs(_height(boxes[mark - 1]) - height) < SEPARATOR_HEIGHT
            ),
            key=lambda mark: boxes[mark - 1][1].start,
        )
        if len(group) != len(ordered) - 1:
            continue
        gaps = [(boxes[m - 1][1].start + boxes[m - 1][1].stop) / 2 for m in group]
        if all(centres[i] < gap < centres[i + 1] for i, gap in enumerate(gaps)):
            return set(group)
    return set()


def _height(box) -> int:
    return box[0].stop - box[0].start


def _panels(sheet: Image.Image, count: int) -> list[np.ndarray]:
    """One boolean mask per step -- the figure and its annotations -- in reading order."""
    pixels = np.asarray(sheet).astype(int)
    labels, found = ndimage.label(
        np.asarray(sheet.convert("L")) < WHITE_CUTOFF, structure=np.ones((3, 3))
    )
    if found < count:
        raise SystemExit(f"sheet has {found} ink components -- fewer than the {count} steps asked for")

    sizes = np.bincount(labels.ravel())
    sizes[0] = 0  # background
    ranked = np.argsort(sizes)[::-1]
    figures, rest = ranked[:count], ranked[count]
    # A figure is a drawn person; the next thing down is a glyph or an arrowhead. If they are the
    # same order of magnitude then the sheet is not what this script assumes -- say so rather than
    # silently writing a caption out as a step figure.
    if sizes[figures[-1]] < 4 * sizes[rest]:
        raise SystemExit(
            f"the {count}th largest component ({sizes[figures[-1]]} px) is not clearly bigger than "
            f"the next ({sizes[rest]} px) -- the panels are not separating cleanly, check the sheet"
        )

    boxes = ndimage.find_objects(labels)
    rows: list[list[int]] = []
    for figure in sorted(figures, key=lambda i: boxes[i - 1][0].start):
        # Same row if it overlaps the row's first figure vertically at all; sheets space their rows
        # far further apart than the figures within a row differ in height.
        if rows and boxes[rows[-1][0] - 1][0].stop > boxes[figure - 1][0].start:
            rows[-1].append(figure)
        else:
            rows.append([figure])
    ordered = [f for row in rows for f in sorted(row, key=lambda i: boxes[i - 1][1].start)]

    marks: dict[int, list[int]] = {figure: [] for figure in ordered}
    accents = []
    for label in range(1, found + 1):
        if label in marks or sizes[label] < MIN_MARK:
            continue
        pick = labels[boxes[label - 1]] == label
        mean = pixels[boxes[label - 1]][pick].mean(axis=0)
        if mean[2] - max(mean[0], mean[1]) < ACCENT_MARGIN:
            continue  # a caption glyph, or a ground shadow
        accents.append(label)

    separators = _separator_row(accents, sizes, boxes, ordered)
    for label in accents:
        if label in separators:
            continue
        for figure in ordered:
            if _over(boxes[label - 1], boxes[figure - 1]):
                marks[figure].append(label)
                break

    print(
        f"{len(rows)} row(s), {len(ordered)} panels, "
        f"{len(separators)} between-panel chevron(s) dropped:"
    )
    panels = []
    for n, figure in enumerate(ordered, 1):
        mask = labels == figure
        for label in marks[figure]:
            mask |= labels == label
        ys, xs = np.nonzero(mask)
        kept = sum(sizes[label] for label in marks[figure])
        print(
            f"  {n}: x {xs.min()}-{xs.max()} y {ys.min()}-{ys.max()}  "
            f"{sizes[figure]} px figure + {len(marks[figure])} annotation(s), {kept} px"
        )
        panels.append(mask)
    return panels


def _cut(sheet: Image.Image, panel: np.ndarray, scale: float) -> Image.Image:
    """One panel on white, everything else on the sheet dropped, enlarged by the set's scale."""
    ys, xs = np.nonzero(panel)
    box = (
        max(0, xs.min() - MARGIN),
        max(0, ys.min() - MARGIN),
        min(sheet.width, xs.max() + 1 + MARGIN),
        min(sheet.height, ys.max() + 1 + MARGIN),
    )

    # White RGB with the panel's own mask as alpha: the neighbours' ink is not merely cropped away,
    # it is dropped pixel by pixel, so a caption inside this panel's box goes too. White underneath
    # means the enlargement's edge pixels blend towards the background, not away.
    cut = Image.new("RGBA", sheet.size, (255, 255, 255, 0))
    cut.paste(sheet, (0, 0), Image.fromarray(panel))
    cut = cut.crop(box)

    size = (max(1, round(cut.width * scale)), max(1, round(cut.height * scale)))
    cut = cut.resize(size, Image.LANCZOS)
    alpha = cut.getchannel("A")
    out = Image.new("RGB", size, (255, 255, 255))
    out.paste(cut.convert("RGB").filter(UNSHARP), (0, 0), alpha)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sheet", type=Path, help="the multi-panel PNG")
    parser.add_argument("--movement", required=True, help="slug for the filenames, e.g. shoulder-bridge")
    parser.add_argument("--steps", type=int, default=5, help="how many panels the sheet holds")
    parser.add_argument("--out", type=Path, default=STEPS_DIR)
    parser.add_argument("--ink-height", type=int, default=INK_HEIGHT)
    args = parser.parse_args()

    sheet = Image.open(args.sheet.expanduser()).convert("RGB")
    panels = _panels(sheet, args.steps)

    heights = [int(np.nonzero(p)[0].max() - np.nonzero(p)[0].min()) + 1 for p in panels]
    scale = args.ink_height / max(heights)
    print(f"shared scale {scale:.3f} from the tallest panel ({max(heights)} px)")

    args.out.mkdir(parents=True, exist_ok=True)
    for n, panel in enumerate(panels, 1):
        path = args.out / f"{args.movement}-{n}.png"
        cut = _cut(sheet, panel, scale)
        cut.save(path)
        print(f"{path.name} {cut.size} {path.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
