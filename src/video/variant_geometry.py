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
