"""What each B1 arm actually shows the model, computed from geometry alone.

``notes/videomae_person_crop_validation_plan.md`` requires every arm to report F3 (how
much of the 224x224 input the athlete occupies), because F3 cannot be fully separated
from F1/F2 and so has to be quantified rather than controlled. This module computes
that, plus the manipulation check the F2 contrast actually rests on: how much of the
person box survives the processor's centre crop.

The pipeline being modelled is ``VideoMAEImageProcessor``, verified against the
checkpoint's own config rather than assumed: ``shortest_edge=224`` resize followed by a
224x224 centre crop. Everything here is exact arithmetic on the box and the frame size,
so it needs no video decoding and no model -- which is also why it belongs in a module
rather than a probe: the Row residual's lesson is that numbers quoted in a write-up
must come from code that can be re-run.

Two results are worth knowing before any accuracy number arrives:

* ``full_frame_letterbox`` is a no-op on the 768 of 1623 videos that are already square
  (47.3%). Their ``full_frame`` input was never centre-cropped, so the arm has nothing
  to restore and the F2 contrast is carried entirely by the other 855.
* ``full_frame_letterbox`` moves F2 and F3 in OPPOSITE directions. Padding a 480x852
  frame to 852x852 restores the cropped-off athlete but shrinks them within the input,
  while ``full_frame``'s centre crop zooms in on whatever it keeps. Two arms scoring
  the same is therefore consistent with an F2 gain cancelling an F3 loss, not only with
  "F2 does not matter".
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from src.video.variant_geometry import CROP_VARIANTS, Box

#: The processor's post-resize centre crop, read from the checkpoint config.
PROCESSOR_SIZE = 224
#: Below this much loss a box counts as fully visible; see ``Framing.truncated``.
SURVIVAL_TOLERANCE = 1e-9

#: Every arm in the B1 matrix, plus the identity arms that share their geometry.
FRAMING_VARIANTS = (
    "full_frame",
    "full_frame_letterbox",
    "person_crop_centercrop",
    "person_crop",
    "background_only",
)


@dataclass(frozen=True)
class Framing:
    """One video's geometry under one arm, after the processor."""

    input_size: tuple[int, int]
    #: Person-box area as a fraction of the 224x224 input. This is F3.
    body_area_fraction: float
    #: Fraction of the person box the centre crop keeps. This is the F2 manipulation.
    box_survival: float
    #: Fraction of the box's HEIGHT lost off the top and off the bottom. Area is the
    #: wrong severity metric for F2: 12% of the box area taken off the bottom edge is
    #: the ankles, and squat depth is measured at the ankles. Which end goes matters
    #: more than how much.
    top_loss: float
    bottom_loss: float
    #: Whether the arm's geometric transform changed the frame at all.
    transformed: bool

    @property
    def truncated(self) -> bool:
        # Tolerance, not paranoia: a box that fits exactly comes out at 1 - 1e-16 from
        # the resize arithmetic, and calling that "truncated" would report every
        # letterboxed video as a truncation and destroy the F2 contrast it measures.
        return self.box_survival < 1.0 - SURVIVAL_TOLERANCE


def variant_input(variant: str, frame_w: int, frame_h: int, box: Box) -> tuple[int, int, Box, bool]:
    """The frame size and person box the processor receives under ``variant``.

    Offsets use the same integer arithmetic as ``letterbox_to_square`` so the modelled
    geometry matches the pixels the extractor produces, not an idealised version.
    """
    if variant in ("full_frame", "reencoded", "background_only"):
        return frame_w, frame_h, box, False

    if variant == "full_frame_letterbox":
        side = max(frame_w, frame_h)
        left = (side - frame_w) // 2
        top = (side - frame_h) // 2
        shifted = Box(box.x0 + left, box.y0 + top, box.x1 + left, box.y1 + top)
        return side, side, shifted, side != frame_w or side != frame_h

    if variant in CROP_VARIANTS:
        box_w = box.x1 - box.x0
        box_h = box.y1 - box.y0
        if variant == "person_crop_centercrop":
            return box_w, box_h, Box(0, 0, box_w, box_h), True
        side = max(box_w, box_h)
        left = (side - box_w) // 2
        top = (side - box_h) // 2
        return side, side, Box(left, top, left + box_w, top + box_h), True

    raise ValueError(f"Unknown variant {variant!r}; expected one of {FRAMING_VARIANTS}.")


def processor_view(
    frame_w: int,
    frame_h: int,
    box: Box,
    size: int = PROCESSOR_SIZE,
) -> tuple[float, float, float, float]:
    """``(body_area_fraction, box_survival, top_loss, bottom_loss)`` after the processor.

    Losses are fractions of the box's own height, so they read directly as "the top
    N% of the athlete is gone".
    """
    if frame_w <= 0 or frame_h <= 0:
        raise ValueError("Frame size must be positive.")

    scale = size / min(frame_w, frame_h)
    scaled_w = frame_w * scale
    scaled_h = frame_h * scale
    crop_left = (scaled_w - size) / 2.0
    crop_top = (scaled_h - size) / 2.0

    box_x0 = box.x0 * scale - crop_left
    box_x1 = box.x1 * scale - crop_left
    box_y0 = box.y0 * scale - crop_top
    box_y1 = box.y1 * scale - crop_top

    visible_w = max(min(box_x1, size) - max(box_x0, 0.0), 0.0)
    visible_h = max(min(box_y1, size) - max(box_y0, 0.0), 0.0)
    visible_area = visible_w * visible_h

    scaled_area = (box.x1 - box.x0) * (box.y1 - box.y0) * scale * scale
    survival = min(visible_area / scaled_area, 1.0) if scaled_area > 0 else 0.0

    scaled_box_h = (box.y1 - box.y0) * scale
    if scaled_box_h > 0:
        top_loss = min(max(-box_y0, 0.0) / scaled_box_h + 0.0, 1.0)
        bottom_loss = min(max(box_y1 - size, 0.0) / scaled_box_h + 0.0, 1.0)
    else:
        top_loss = bottom_loss = 0.0
    return visible_area / (size * size), survival, top_loss, bottom_loss


def frame_variant(variant: str, frame_w: int, frame_h: int, box: Box) -> Framing:
    input_w, input_h, shifted, transformed = variant_input(variant, frame_w, frame_h, box)
    area_fraction, survival, top_loss, bottom_loss = processor_view(input_w, input_h, shifted)
    return Framing(
        input_size=(input_w, input_h),
        body_area_fraction=area_fraction,
        box_survival=survival,
        top_loss=top_loss,
        bottom_loss=bottom_loss,
        transformed=transformed,
    )


#: The subsets B1 reads its F2 contrast on. ``full_frame`` vs ``full_frame_letterbox``
#: cannot differ at all on a square video, and cannot differ in body COMPLETENESS on a
#: video the centre crop never truncated -- so the marginal over all 1623 dilutes the
#: manipulation by videos it does not touch. Pre-register which subset is primary.
SUBSETS = ("all", "non_square", "truncated_by_full_frame")


def select_rows(rows: list[dict], subset: str = "all") -> list[dict]:
    usable = [row for row in rows if row.get("box")]
    if subset == "all":
        return usable
    if subset == "non_square":
        return [row for row in usable if row["frame_size"][0] != row["frame_size"][1]]
    if subset == "truncated_by_full_frame":
        return [
            row
            for row in usable
            if frame_variant("full_frame", int(row["frame_size"][0]), int(row["frame_size"][1]), Box(*row["box"])).truncated
        ]
    raise ValueError(f"Unknown subset {subset!r}; expected one of {SUBSETS}.")


def truncation_cause(frame_w: int, frame_h: int, box: Box) -> str:
    """Why ``full_frame`` loses part of this athlete: ``scale``, ``framing`` or ``none``.

    The processor's shortest-edge resize plus square centre crop is exactly "keep the
    centred ``min(W, H)``-square of the original frame". So a box can be lost two ways,
    and they have different fixes: it is TALLER than that window (scale -- only zooming
    out helps, which costs F3), or it fits but sits away from the frame's centre
    (framing -- re-centring the window on the athlete costs no zoom at all). Half of
    the 613 truncated Fitness-AQA squats are the second kind, which is why the F2
    manipulation does not have to cost the 30% body area ``full_frame_letterbox`` pays.
    """
    window = min(frame_w, frame_h)
    if not frame_variant("full_frame", frame_w, frame_h, box).truncated:
        return "none"
    return "scale" if (box.y1 - box.y0) > window else "framing"


def split_counts(rows: list[dict], split_map: dict[str, str], subset: str = "all") -> dict[str, int]:
    """How many videos of ``subset`` land in each split.

    A contrast is only as powerful as its TEST videos, and the subsets B1 needs are a
    minority of a 244-video test split. Reporting the corpus count alone reads as more
    evidence than the comparison can carry.
    """
    counts: dict[str, int] = {}
    for row in select_rows(rows, subset):
        split_name = split_map.get(str(row["video_id"]), "unassigned")
        counts[split_name] = counts.get(split_name, 0) + 1
    return counts


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p10": 0.0, "median": 0.0, "p90": 0.0}
    ordered = sorted(values)
    last = len(ordered) - 1
    return {
        "p10": ordered[int(round(0.1 * last))],
        "median": median(ordered),
        "p90": ordered[int(round(0.9 * last))],
    }


def summarize_manifest(
    rows: list[dict],
    variants: tuple[str, ...] = FRAMING_VARIANTS,
    subset: str = "all",
) -> dict[str, dict]:
    """Per-arm F3 and centre-crop survival over a ``build_video_variants`` manifest.

    Rows must come from a ``person_crop`` manifest: both crop arms take the expanded
    box, and tracking that same box through every arm is what makes the arms
    comparable. Rows whose box is null (no person ever visible) are counted and
    excluded -- their arms are untransformed by construction.
    """
    usable = select_rows(rows, subset)
    summary: dict[str, dict] = {
        "subset": subset,
        "n_rows": len(rows),
        "n_selected": len(usable),
        "n_boxless": len(rows) - len(select_rows(rows, "all")),
        "arms": {},
    }

    for variant in variants:
        areas: list[float] = []
        survivals: list[float] = []
        top_losses: list[float] = []
        bottom_losses: list[float] = []
        truncated = 0
        transformed = 0
        for row in usable:
            frame_w, frame_h = (int(value) for value in row["frame_size"])
            framing = frame_variant(variant, frame_w, frame_h, Box(*row["box"]))
            areas.append(framing.body_area_fraction)
            survivals.append(framing.box_survival)
            top_losses.append(framing.top_loss)
            bottom_losses.append(framing.bottom_loss)
            truncated += framing.truncated
            transformed += framing.transformed

        summary["arms"][variant] = {
            "body_area_fraction": _percentiles(areas),
            "box_survival": _percentiles(survivals),
            "top_loss": _percentiles(top_losses),
            "bottom_loss": _percentiles(bottom_losses),
            "n_truncated": truncated,
            "fraction_truncated": truncated / len(usable) if usable else 0.0,
            "n_transformed": transformed,
            "n_identical_to_source": len(usable) - transformed,
        }

    return summary
