"""Cut a common-mistake sheet into the wrong / correct pair the movement detail page shows.

Third sibling of prep_movement_art.py (library cards) and prep_movement_detail_art.py (step
figures + muscle plates), and here for their reason: the WebPs are hand-supplied illustrations with
specific surgery applied, and committing the binaries without the recipe leaves nobody able to redo
it for the next of the 80 pairs `frontend/public/movements/mistakes/README.md` lists.

The sources arrive as ONE sheet per fault -- two cards side by side, the fault drawn on the left
and the same body doing it right on the right -- rather than as two files. This cuts them:

    frontend/public/movements/mistakes/<sheet>.png   ->  <fault-slug>-wrong.webp
                                                         <fault-slug>-correct.webp

The sheet names describe the drawing ("squat-knee-caving-in") and the outputs are named for the
detector's `fault_id` ("knees-inward"), which is why SHEETS below is a table and not a glob: the
two namings do not derive from each other, and guessing is how a pair ends up on the wrong card.

Four things the conversion does, two shared with its siblings and two that are only ever a problem
here:

  * SPLIT between the two drawn figures -- found as the two largest masses of ink, then cut at the
    column inside the gap that strands the least annotation -- rather than at the midpoint, so a
    sheet whose panels are not exactly equal does not lose a limb and a sheet whose panels nearly
    touch does not leave one half's verdict sitting in the other's. See _band and _gutter, which
    each say which sheet forced them. Two panels is the only shape this understands.
  * KNOCK OUT everything neutral and reachable from the panel's border: the page, the card plate
    and the ground shadow, leaving the drawing and its arrows. This has to run BEFORE the crop,
    not after: the cards are drawn on a plate darker than any ink cutoff worth having, so a
    bounding box taken first is the box of the card, not of the figure. The panel this lands in is
    tinted (`bg-danger/[0.06]` / `bg-secondary/[0.07]`), so an opaque plate would render as a
    visible rectangle -- the same reason the step figures are knocked out.
  * PUT BACK what the knock-out cuts adrift, which on these sheets is the white socks and, with
    them, the shoes: they are the page's tone exactly and they are open to the page at the ankle,
    so no threshold can keep them and the shoe ends up floating below the leg. See _reattached.
  * FIT BOTH HALVES INTO ONE BOX: shared rows, and a shared width centred on each half's own ink.
    The pair is the whole point of the picture, and `object-contain` scales each panel
    independently: crop each half to its own ink and the wider one is rendered SMALLER, so the same
    body looks bigger on one side than the other and the reader reads a size difference the drawing
    never claimed. The annotations guarantee the widths differ -- on the knee sheet the wrong
    panel's arrows point inward and the correct panel's point outward, and outward is wider. One
    size, one scale, one body. See split_sheet for why rows are shared outright and columns are not.

The pair is then checked before it is written, and checked on GREEN -- see split_sheet for why
red cannot decide it. A left/right swap survives every other check here and both vitest and the
roster test (all of which assert only that the files EXIST), and it ships a picture that lies --
the one thing the page refuses to do. The annotations are the only thing in the drawing that knows
which half it is, which is why DOMINANCE is fussier than it looks.

What NOTHING here can check is that a sheet is mapped to the right fault: both halves come out
correctly coloured and the pair simply lands on the wrong card. SHEETS is the only thing standing
in the way, so keep reading it against the authored titles in movementMistakes.ts, and look at the
result.

Needs Pillow, numpy and scipy, all three already in .venv and none of them in requirements.txt --
the same footing as split_step_sheet.py's. This runs offline against a file on disk; nothing it
needs belongs in the web image.

Usage (from the repo root):
    .venv\\Scripts\\python.exe scripts/prep_mistake_art.py
    .venv\\Scripts\\python.exe scripts/prep_mistake_art.py --preview   # also write the tinted
                                                                      # composites to check by eye
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[1]
MISTAKES_DIR = REPO_ROOT / "frontend" / "public" / "movements" / "mistakes"

# source sheet -> the fault's slug, i.e. its `fault_id` hyphenated, exactly as movementMistakes.ts's
# `art()` builds it. Add a line per sheet dropped into the directory.
SHEETS: dict[str, str] = {
    "squat-knee-caving-in.png": "knees-inward",
    "squat-knees-travelling-too-far.png": "knees-forward",
    "squat-not-squatting-deep-enough.png": "shallow-depth",
    "squat-leaning-too-far-forward.png": "excessive-forward-lean",
    "squat-heels-lifting-off-the-floor.png": "heel-rise",
    "overhead-press-not-locking-out-at-the-top.png": "ohp-incomplete-lockout",
    "overhead-press-leaning-back-and-arching-the-lower-back.png": "ohp-lumbar-hyperextension",
    "overhead-press-one-arm-leading-the-press.png": "ohp-asymmetric-press",
    "overhead-press-not-pressing-fully-overhead.png": "ohp-insufficient-elevation",
    "overhead-press-head-poking-forward-at-lockout.png": "ohp-forward-head",
    "push-up-hips-sagging.png": "pushup-hip-sag",
    "push-up-not-going-deep-enough.png": "pushup-shallow-depth",
    "push-up-head-dropping-forward.png": "pushup-head-drop",
    "push-up-elbows-flaring-out.png": "pushup-elbow-flare",
    "lunge-not-lunging-deep-enough.png": "lunge-insufficient-depth",
    "lunge-front-knee-sliding-past-the-toes.png": "lunge-knee-past-toes",
    "lunge-front-knee-caving-inward.png": "lunge-knee-valgus",
    "lunge-hip-dropping-on-the-free-side.png": "lunge-pelvic-drop",
    "deadlift-rounding-the-lower-back.png": "deadlift-lumbar-flexion",
    "row-torso-rising-out-of-the-hinge.png": "row-torso-rising",
    "row-not-completing-the-pull.png": "row-incomplete-rom",
    "row-using-momentum.png": "row-momentum-jerk",
    "row-one-side-pulling-harder.png": "row-asymmetric-pull",
    "deadlift-not-finishing-the-lockout.png": "deadlift-incomplete-lockout",
    "deadlift-hips-shooting-up-first.png": "deadlift-hips-shoot-up",
    "shoulder-bridge-hips-not-reaching-the-top.png": "bridge-incomplete-hip-extension",
    "leg-abduction-leaning-the-trunk-to-lift-the-leg.png": "abd-pelvic-drop-trunk-lean",
    # The first of these three is misspelt at source ("aparts-hrugging"); the table is explicit
    # precisely so a sheet's name never has to be trusted or corrected.
    "band-pull-aparts-hrugging-the-shoulders.png": "bpa-shrugging",
    "band-pull-aparts-not-spreading-the-hands-fully.png": "bpa-incomplete-rom",
    "band-pull-aparts-leaning-back-to-open-the-band.png": "bpa-trunk-extension-compensation",
    "bicep-curl-elbows-drifting-forward.png": "curl-elbow-drift-forward",
    "bicep-curl-swinging-the-body.png": "curl-trunk-swing-momentum",
    "bicep-curl-half-reps.png": "curl-incomplete-rom",
    "torso-twist-losing-the-braced-torso.png": "tt-trunk-not-braced",
}

# A gap narrower than this between the two figures is not something to split on.
MIN_GUTTER = 8
# Breathing room kept around the art, in source pixels.
PAD = 12
# How far a pixel's channels may spread and still count as neutral rather than as part of the
# drawing. The violet kit, the skin and the arrows all clear this by a wide margin; the plate's
# grain does not.
GREY_TOLERANCE = 14
# The darkest neutral still treated as background. Set below the ground shadow's core (~200) and
# far above the drawing's own neutrals -- black shorts, black hair -- so the shadow goes and they
# stay. The white socks are lighter than this and survive on reachability, not on tone.
GREY_FLOOR = 176

# How far ahead of BOTH other channels a pixel has to sit to count as an annotation mark rather
# than as the drawing, per channel -- red then green. The two numbers are far apart because the
# palette is not symmetric: the drawing contains a lot of red (skin) and no green at all.
#
#   RED has to clear skin, which is the one thing here that is genuinely red-dominant, and it does
#   not clear it by much: flat skin sits 46-62 ahead of its brightest other channel, but the dark
#   strokes the skin is CONTOURED with reach 131, against the red arrows' 137-167. Two things keep
#   that from being as tight as it looks. The strokes are two pixels wide and _has runs on the
#   panel AFTER it has been resampled down by about a third, which blends a 2px stroke into the
#   skin and leaves a 12px arrow an arrow; and the count floor below then wants a hundred of them.
#   Measured that way the correct halves carry 0-2 red pixels, not 100. Lower this and the squat
#   sheets start reading as red on BOTH sides, which does not fail loudly -- it makes the swap
#   check vacuous, which is how it was first written and why it is measured now.
#   GREEN has to clear nothing: no skin tone, no violet kit, no black short or white sock is
#   green-dominant at all. It only has to be low enough for the PALEST green any sheet uses, and
#   the sheets disagree about that by a factor of two -- squat's arrow green is (33,180,81), 99
#   ahead, while the push-up sheets draw (107,184,74) at 77 and (133,179,111) at only 46. A single
#   number tuned on squat finds ZERO pixels of the push-up greens, which does not read as "paler
#   green" downstream, it reads as "this half carries no green, so the pair is swapped".
DOMINANCE = (110, 40)
# A piece of drawing this big that the knock-out has cut adrift is a limb, not a speck: put it
# back. Measured -- the severed shoes run 3774-8536 px, and the largest thing on any sheet that is
# adrift on PURPOSE (a dash of an annotation line) is 2185.
REATTACH_MIN = 3000
# How wide a neck of background the reattachment may close over. The sock bands measured up to 40
# px across their mouths, so a disk that cannot pass through 50 covers them.
SEAM_RADIUS = 25

# And how many such pixels make a half "annotated". The finished halves carry 267-1538 of their own
# colour; this is the floor below which a scatter of resampled edge pixels would pass for an arrow.
MIN_MARK_PIXELS = 100
# How far the correct half's green has to outrun the wrong half's. See split_sheet for why the
# orientation is decided on GREEN and only corroborated by red.
MARK_MARGIN = 3

# Rendered size, x3 for a high-DPI screen. MistakePanel is `h-[152px]`, and its width comes from
# the pair's wrapper: `max-w-[296px] grid-cols-2 gap-1.5`, so (296 - 6) / 2 = 145 CSS px, and at xl
# the column it sits in is a fixed 296px too. Both numbers are the panel's real size rather than an
# estimate, which matters now that one of them binds: the art is `object-contain` inside it. Both numbers are needed, not just the height: `object-contain`
# fits the LONG side, so which one binds depends on the drawing. Standing art (every squat sheet)
# is taller than it is wide and fills the height; the push-up sheets are the opposite -- a prone
# figure roughly 2.4:1 -- and matching its height would render it 145 wide on the page anyway,
# after exporting 1100 px of it. Fit inside the box and the binding side is always the drawn one.
# Same x3 convention as prep_movement_detail_art.py's STEP_HEIGHT.
PANEL_WIDTH = 435
PANEL_HEIGHT = 456

# The two panel tints, for --preview only: `bg-danger/[0.06]` and `bg-secondary/[0.07]` over the
# white surface, resolved from tailwind.config.js's danger #ff5a5a and secondary #22c55e.
PREVIEW_TINT = {"wrong": (255, 240, 240), "correct": (243, 251, 246)}
PREVIEW_SIZE = (145, 152)


def _flatten(path: Path) -> Image.Image:
    """The sheet as opaque RGB. Composited onto white FIRST because a source carrying an alpha
    channel would otherwise defeat both the gutter search and the knock-out."""
    image = Image.open(path).convert("RGBA")
    flat = Image.new("RGB", image.size, (255, 255, 255))
    flat.paste(image, mask=image.split()[3])
    return flat


def _band(knocked: Image.Image, name: str) -> tuple[int, int]:
    """The candidate split columns: the clear span between the two drawn figures.

    The figures are the two largest connected masses of surviving ink. What has to be checked is
    not that they DWARF everything else -- an earlier version demanded a 10x margin over the third
    mass, on the evidence that the squat bodies run 121k-149k px against a third place of 2.2k, and
    the bicep curl sheet is a counter-example rather than a malformed sheet: it draws two magnified
    circular insets per panel, at 42k each, and they are as much a part of their panel as the body
    is. What has to hold is that the two anchors are ONE PER PANEL, which is asserted directly.

    Deliberately NOT "the emptiest run of columns", which was the first rule here and is wrong on
    the shallow depth sheet: its two panels share one grey dashed floor line spanning the whole
    sheet, so no column between them is ever empty, and a threshold loose enough to see past the
    floor line also reads the correct panel's dashed hip line as empty space and cuts through it.
    """
    mask = np.asarray(knocked.split()[3]) > 0
    labels, count = ndimage.label(mask, structure=np.ones((3, 3)))
    if count < 2:
        raise SystemExit(f"{name}: found {count} ink mass(es) -- a two-panel sheet has two figures")
    sizes = ndimage.sum(mask, labels, range(1, count + 1))
    ranked = np.argsort(sizes)[::-1]

    columns = [np.nonzero(labels == label + 1)[1] for label in ranked[:2]]
    left, right = sorted(columns, key=lambda c: c.min())
    midline = knocked.width / 2
    if not left.mean() < midline <= right.mean():
        raise SystemExit(
            f"{name}: the two largest ink masses are centred at x={left.mean():.0f} and "
            f"x={right.mean():.0f} on a sheet {knocked.width} wide, so they are not one figure per "
            f"panel -- refusing to guess where the panels divide"
        )
    lo, hi = int(left.max()) + 1, int(right.min())
    if hi - lo < MIN_GUTTER:
        raise SystemExit(
            f"{name}: only {hi - lo}px separates the two figures -- too little to split on"
        )
    return lo, hi


def _gutter(sheet_rgb: Image.Image, band: tuple[int, int]) -> int:
    """The x to split at, chosen inside `band` to strand as little annotation as possible.

    Any column in the band separates the two bodies, so what is left to decide is where the marks
    go, and one sheet makes that a real decision rather than a formality. On the shallow depth
    sheet the correct panel's green dashed hip line reaches LEFT past its own figure and stops one
    pixel short of the wrong figure's fingertips (green from x=598, left body to x=599) -- so the
    band's midpoint leaves a green fragment sitting in the wrong panel, a mark of the other half's
    verdict inside this one. Costing each candidate by exactly that -- green stranded left of the
    cut plus red stranded right of it -- puts the split hard against the body instead, and on the
    four sheets with nothing to strand every column ties at zero and this is the band's midpoint.
    """
    pixels = np.asarray(sheet_rgb, dtype=np.int16)
    green = _marks(pixels, 1).sum(axis=0)
    red = _marks(pixels, 0).sum(axis=0)
    # Splitting at x puts columns < x in the wrong half and the rest in the correct one.
    green_stranded = np.cumsum(green) - green
    red_stranded = red.sum() - (np.cumsum(red) - red)
    cost = (green_stranded + red_stranded)[band[0] : band[1] + 1]

    cheapest = np.flatnonzero(cost == cost.min()) + band[0]
    # The cheapest columns are contiguous in practice; take the middle of the longest such run so a
    # sheet with nothing to strand still splits down the middle of its gap.
    runs = np.split(cheapest, np.flatnonzero(np.diff(cheapest) > 1) + 1)
    longest = max(runs, key=len)
    return int((longest[0] + longest[-1] + 1) // 2)


def _knocked_out(panel: Image.Image) -> Image.Image:
    """One half with the page, the card plate and the ground shadow gone, the drawing kept.

    Background is "grey and light" -- unsaturated, and no darker than GREY_FLOOR -- AND reachable
    from the panel's border. Both halves of that carry weight:

      * GREY rather than near-white, because the card is not the only neutral thing behind the
        figure. The ground shadow is an ellipse fading from ~250 to ~200, and a near-white test
        keeps its core: a grey smudge under the feet on a tinted panel, where the step figures
        beside it stand on nothing. Colour is what separates it from the drawing, which is violet
        kit, skin and saturated arrows; the only neutral things IN the drawing are the black shorts
        and hair (far below the floor) and the white socks (enclosed).
      * REACHABLE, so the enclosed whites survive -- the highlights inside the shoes are the plate's
        own tone and are kept for exactly the reason `prep_movement_detail_art` gives: only
        background CONNECTED TO THE BORDER goes. The socks are the case where that is not enough.
        Whether a sock is enclosed is an accident of the pose: the squat five enclose theirs, three
        other sheets leave a sock open to the page at the ankle and lose it, and the shoe with it.
        _reattached below is the second half of this rule and the reason that is not what ships.

    A plain flood fill was tried first and is not enough on these sheets. The plate carries grain
    that straddles a tight threshold, which breaks the fill's connectivity and leaves the panel
    speckled with islands the fill could not reach; a per-pixel test with room in it, flooded as a
    mask, has no such seam.
    """
    pixels = np.asarray(panel, dtype=np.int16)
    lightest = pixels.max(axis=2)
    darkest = pixels.min(axis=2)
    background = (lightest - darkest <= GREY_TOLERANCE) & (darkest >= GREY_FLOOR)

    labels, _ = ndimage.label(background)
    edge = np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])
    outside = np.unique(edge[edge > 0])
    alpha = np.where(np.isin(labels, outside), 0, 255).astype(np.uint8)

    rgba = panel.convert("RGBA")
    rgba.putalpha(Image.fromarray(alpha, "L"))
    return _reattached(rgba)


def _reattached(rgba: Image.Image) -> Image.Image:
    """Put back the white socks, which the knock-out is right to erase and wrong to keep erased.

    The socks are the one part of the drawing the background rule cannot see as drawing: they are
    the same tone as the page (measured 249-255 against a 250 page, i.e. no separation AT ALL, so
    no threshold anywhere can fix this), and unlike the enclosed whites they are open to the
    background on both sides of the ankle. So they go, and the shoe goes with them -- not erased,
    but cut adrift, floating a visible gap below the leg on the shipped panel. Three sheets do it
    (both overhead-press figures, two push-up ones); the squat five happen to enclose their socks
    and are untouched by this.

    What identifies a sock is not its colour but that closing a narrow neck of background REJOINS
    TWO PIECES OF ONE DRAWING. So: close the ink by a disk, and keep only what the closing adds
    BETWEEN a figure and a big adrift piece -- within reach of both. Everything else the closing
    would fill -- the channel between the dumbbell and the head, the gaps between fingers -- has
    the same piece of drawing on both sides and is dropped, which matters because the pixels
    restored here are restored AS THEY ARE IN THE SOURCE: a sock comes back white, but a filled
    channel would come back page-grey, an opaque smear on a tinted panel.

    Adrift ANNOTATION is left adrift, by the size floor and by the mark test. The dashes of a
    dashed line are exactly this shape -- pieces of one drawing separated by a short neck -- and
    joining them up would redraw a reference line as a solid one.
    """
    alpha = np.asarray(rgba.split()[3])
    ink = alpha > 0
    labels, count = ndimage.label(ink, structure=np.ones((3, 3)))
    if count < 3:
        return rgba
    sizes = ndimage.sum(ink, labels, range(1, count + 1))
    ranked = np.argsort(sizes)[::-1]
    figures = {int(ranked[0]) + 1, int(ranked[1]) + 1}

    pixels = np.asarray(rgba, dtype=np.int16)
    annotation = _marks(pixels, 0) | _marks(pixels, 1)
    adrift: set[int] = set()
    for label in ranked[2:]:
        if sizes[label] < REATTACH_MIN:
            break  # ranked descending, so nothing further down qualifies either
        piece = labels == label + 1
        if (annotation & piece).sum() > sizes[label] * 0.25:
            continue
        adrift.add(int(label) + 1)
    if not adrift:
        return rgba

    # A binary closing, as two distance transforms: dilate by SEAM_RADIUS, erode by the same. What
    # it adds is every pocket of background too narrow for a disk that size to sit in.
    near = ndimage.distance_transform_edt(~ink) <= SEAM_RADIUS
    closed = ndimage.distance_transform_edt(near) > SEAM_RADIUS

    # ... of which we want only what lies BETWEEN the two pieces, so: also within reach of both.
    # The closing alone is not enough of a filter. Its additions around one figure are a single
    # connected ribbon -- every concavity in the silhouette fills, and the fills join up around the
    # body -- so asking "does this patch touch a figure and an adrift piece?" answers yes for a
    # ribbon that reaches the ankle at one end and, at the other, slabs the background behind the
    # shoulders. Reach from both pieces has no such loophole and needs no patch labelling.
    keep = (
        closed
        & ~ink
        & (ndimage.distance_transform_edt(~np.isin(labels, sorted(adrift))) <= SEAM_RADIUS)
        & (ndimage.distance_transform_edt(~np.isin(labels, sorted(figures))) <= SEAM_RADIUS)
    )
    if not keep.any():
        return rgba

    restored = rgba.copy()
    restored.putalpha(Image.fromarray(np.where(keep, 255, alpha).astype(np.uint8), "L"))
    return restored


def _ink_box(rgba: Image.Image, name: str) -> tuple[int, int, int, int]:
    """The padded box around what survived the knock-out.

    Deliberately NOT clamped to the panel. A box that runs off an edge is cropped off the edge, and
    `Image.crop` pads an RGBA with transparency when asked for pixels that are not there -- which is
    the right answer for art that is about to be composited anyway, and saves the padding being
    silently thinner on whichever side the drawing sits closest to.
    """
    box = rgba.split()[3].getbbox()
    if box is None:
        raise SystemExit(f"{name}: the knock-out erased the whole panel")
    return (box[0] - PAD, box[1] - PAD, box[2] + PAD, box[3] + PAD)


def _window(box: tuple[int, int, int, int], width: int) -> int:
    """The left edge of a `width`-wide column centred on `box`."""
    return round((box[0] + box[2]) / 2 - width / 2)


def _marks(pixels: np.ndarray, channel: int) -> np.ndarray:
    """Mask of the annotation marks on `channel` (0 red, 1 green) in an RGB-or-RGBA pixel array."""
    bands = pixels[..., :3]
    mine = bands[..., channel]
    others = np.delete(bands, channel, axis=-1).max(axis=-1)
    return (mine > 120) & (mine - others > DOMINANCE[channel])


def _mark_pixels(rgba: Image.Image, channel: int) -> int:
    """How many annotation-mark pixels on `channel` (0 red, 1 green) the panel actually shows.

    Counted on the finished panel rather than on the source, because that is the picture that
    ships and because the resample is doing real work here -- see DOMINANCE for why a 2px contour
    stroke and a 12px arrow are not as far apart in colour as they look.
    """
    pixels = np.asarray(rgba, dtype=np.int16)
    visible = pixels[..., 3] >= 200
    return int((_marks(pixels, channel) & visible).sum())


def _preview(rgba: Image.Image, tone: str, path: Path) -> None:
    """The panel as the page will draw it: `object-contain` inside 145x152 over the tint."""
    plate = Image.new("RGB", PREVIEW_SIZE, PREVIEW_TINT[tone])
    art = rgba.copy()
    art.thumbnail(PREVIEW_SIZE, Image.LANCZOS)
    plate.paste(
        art,
        ((PREVIEW_SIZE[0] - art.width) // 2, (PREVIEW_SIZE[1] - art.height) // 2),
        mask=art.split()[3],
    )
    plate.save(path)


def split_sheet(sheet: Path, slug: str, preview: bool) -> None:
    flat = _flatten(sheet)
    knocked = _knocked_out(flat)
    x = _gutter(flat, _band(knocked, sheet.name))
    halves = {
        "wrong": knocked.crop((0, 0, x, knocked.height)),
        "correct": knocked.crop((x, 0, knocked.width, knocked.height)),
    }
    boxes = {tone: _ink_box(rgba, f"{sheet.name} ({tone})") for tone, rgba in halves.items()}

    # ONE crop size for both halves, so `object-contain` lands both figures at one scale (see the
    # module docstring) -- but the two axes get there differently.
    #
    # ROWS ARE SHARED OUTRIGHT. Both halves are cut from the same rows of the same sheet, so taking
    # the union y-range and applying it verbatim keeps whatever the drawing aligned. The shallow
    # depth sheet is the one that proves this has to be so: its two figures stand on a single grey
    # dashed floor line, and centring each half on its own ink would put that floor at two
    # different heights in the pair. It would also flatten the drawing's whole point -- the right
    # figure's head starts ~100 px lower BECAUSE he is squatting deeper, and that gap is the
    # comparison, not slack to be trimmed away.
    #
    # COLUMNS ARE CENTRED PER HALF, because the horizontal is where the halves genuinely differ:
    # the lean sheet's red arc reaches well left of its figure and the correct half has no such
    # mark, so a shared x-range would hang the right figure off to one side. Same width, own
    # centre. Overhang past the sheet crops to transparency, which costs nothing here.
    top = min(box[1] for box in boxes.values())
    bottom = max(box[3] for box in boxes.values())
    width = max(box[2] - box[0] for box in boxes.values())

    cut: dict[str, Image.Image] = {}
    for tone, rgba in halves.items():
        left = _window(boxes[tone], width)
        art = rgba.crop((left, top, left + width, bottom))
        # Both halves were cut to the same width and the same rows, so this scale is the same
        # number twice -- which is the point, and is asserted below rather than trusted.
        scale = min(PANEL_WIDTH / art.width, PANEL_HEIGHT / art.height)
        cut[tone] = art.resize((round(art.width * scale), round(art.height * scale)), Image.LANCZOS)

    if cut["wrong"].size != cut["correct"].size:
        raise SystemExit(f"{sheet.name}: the halves came out different sizes -- {cut['wrong'].size} "
                         f"vs {cut['correct'].size} -- so the page would draw one body bigger")

    # WHICH HALF IS WHICH IS DECIDED ON GREEN. Red cannot decide it, and the band pull apart sheets
    # are why: they draw the resistance band ITSELF in red, so a correct half carries ~1000 red
    # pixels of equipment before any arrow is drawn, and on the incomplete-rom sheet the correct
    # half carries MORE red than the wrong one (1011 vs 619) because a fully spread band is longer
    # than a short one. Every rule of the form "the correct half is not red" or "the wrong half is
    # redder" gets that sheet wrong, and gets it wrong in the direction that refuses good art.
    #
    # Green has no such problem, and it is the same asymmetry that splits DOMINANCE: this palette
    # contains a great deal of red -- skin, contour strokes, and now equipment -- and contains no
    # green whatsoever. Measured across all 34 sheets the wrong half carries 0-2 green pixels and
    # the correct half 175-1446, without a single exception. So green exclusivity decides it, and
    # it catches a swap from either side: swap the halves and the correct one is left with the 0.
    #
    # Red is kept as a presence check only -- a wrong half with no red marks at all is a sheet that
    # is not annotated the way this expects, whatever else is true of it.
    red_wrong = _mark_pixels(cut["wrong"], 0)
    green_wrong = _mark_pixels(cut["wrong"], 1)
    green_correct = _mark_pixels(cut["correct"], 1)
    if green_correct < MIN_MARK_PIXELS or green_wrong * MARK_MARGIN > green_correct:
        raise SystemExit(
            f"{sheet.name}: green is not where it should be -- {green_correct} px in the correct "
            f"half against {green_wrong} in the wrong one. The halves are probably swapped, and a "
            f"mislabelled pair is worse than no pair"
        )
    if red_wrong < MIN_MARK_PIXELS:
        raise SystemExit(
            f"{sheet.name}: the wrong half carries only {red_wrong} red px, so it is not annotated "
            f"the way this expects -- refusing to write a pair it cannot check"
        )

    for tone, art in cut.items():
        out = MISTAKES_DIR / f"{slug}-{tone}.webp"
        art.save(out, "WEBP", quality=88, method=6)
        print(f"{sheet.name} [{tone}] -> {out.name}  {art.width}x{art.height}  {out.stat().st_size // 1024} KB")
        if preview:
            _preview(art, tone, MISTAKES_DIR / f"{slug}-{tone}.preview.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--preview",
        action="store_true",
        help="also write <slug>-<tone>.preview.png: the panel over its tint, at the page's size",
    )
    args = parser.parse_args()

    for name, slug in SHEETS.items():
        sheet = MISTAKES_DIR / name
        if not sheet.exists():
            print(f"skipped {name}: not in {MISTAKES_DIR.relative_to(REPO_ROOT)} (sources are gitignored)")
            continue
        split_sheet(sheet, slug, args.preview)


if __name__ == "__main__":
    main()
