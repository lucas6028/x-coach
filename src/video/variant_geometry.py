"""Box and variant names for the VideoMAE arms -- no OpenCV, no torch.

Split out of ``squat_video_variants`` so that everything which only reasons about
geometry (which arm takes a box, how much of the athlete a crop keeps) stays importable
in the torch-free CI suite. The pixel work still lives next door and still owns the
decode/encode path; this module owns only the vocabulary and the rectangle.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Every arm ``build_video_variants`` can build. ``full_frame`` is absent by design:
#: it is the untouched source, so it is a choice for the extractor, not the builder.
VARIANTS = (
    "person_crop",
    "person_crop_centercrop",
    "background_only",
    "full_frame_letterbox",
    "reencoded",
)
#: Variants whose transform is a function of the person box, so the extractor must be
#: given a manifest. ``full_frame_letterbox`` and ``reencoded`` are deliberately absent:
#: they are box-free, and requiring a manifest for them would invite a null box to be
#: read as "leave untouched".
BOX_VARIANTS = ("person_crop", "person_crop_centercrop", "background_only")
#: Both crop arms take the SAME expanded box; the letterbox is their only difference.
CROP_VARIANTS = ("person_crop", "person_crop_centercrop")

DEFAULT_VISIBILITY = 0.5
DEFAULT_MARGIN = 0.15
#: Neutral grey for the letterbox -- the usual detection-pipeline value, mid-range so
#: the padding is neither a bright nor a black frame around the athlete.
LETTERBOX_FILL = 114


@dataclass(frozen=True)
class Box:
    x0: int
    y0: int
    x1: int
    y1: int

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x0, self.y0, self.x1, self.y1)


def expand_box(box: Box, width: int, height: int, margin: float = DEFAULT_MARGIN) -> Box:
    """Grow ``box`` by ``margin`` on each side, clamped inside the frame.

    The margin is slack for landmarks the visibility threshold drops, not for body
    parts the model does not emit: MediaPipe pose JSON carries all 33 landmarks,
    including the hands (17-22) and the heels and foot tips (29-32), so a
    landmark-tight box already reaches the extremities. Measured over 120 videos, the
    visible-only box misses a median -0.3% of the all-landmark box's area and more
    than 10% of it in only 8.3% of videos. Aspect ratio is left alone -- squaring
    happens later as padding, not as more background.

    Lives here rather than beside the pixel work because it is arithmetic on a
    rectangle: REHAB24-6's box builder needs it without pulling in OpenCV, and this
    module is the half of the variant vocabulary that stays importable in the lean CI
    suite. ``squat_video_variants`` re-exports it, so existing imports are unaffected.
    """
    pad_x = (box.x1 - box.x0) * margin / 2.0
    pad_y = (box.y1 - box.y0) * margin / 2.0
    x0 = max(int(round(box.x0 - pad_x)), 0)
    y0 = max(int(round(box.y0 - pad_y)), 0)
    x1 = min(int(round(box.x1 + pad_x)), width)
    y1 = min(int(round(box.y1 + pad_y)), height)
    return Box(x0, y0, max(x1, x0 + 1), max(y1, y0 + 1))
