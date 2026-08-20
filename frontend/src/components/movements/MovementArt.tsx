import MovementIcon from "./MovementIcon";

// The illustration shown on a movement card, keyed by the canonical English movement name.
//
// The first eight (squat through jumping jacks) started as the exercise_library_muse-spark
// reference's own figures; `scripts/prep_movement_art.py` documents that original knockout-to-
// transparency pipeline. What ships today, for all sixteen, are opaque 1254x1254 pre-matted
// squares -- see the "stage" comment in MovementCard.tsx for why full-bleed opaque art replaced
// the trimmed transparent cutouts.
export const ART: Record<string, string> = {
  Squat: "squat.png",
  Lunge: "lunge.png",
  Deadlift: "deadlift.png",
  "Leg Abduction": "leg-abduction.png",
  "Shoulder Bridge": "shoulder-bridge.png",
  "Push-up": "pushup.png",
  "Overhead Press": "overhead-press.png",
  Row: "row.png",
  "Bicep Curl": "bicep-curl.png",
  "Band Pull Apart": "band-pull-apart.png",
  "Arm Abduction": "arm-abduction.png",
  "Arm VW": "arm-vw.png",
  "Sit-up": "situp.png",
  "Torso Twist": "torso-twist.png",
  "Jumping Jacks": "jumping-jack.png",
  "High Knee": "high-knee.png",
};

interface Props {
  /** Canonical English movement name — the key, not the label the reader sees. */
  movement: string;
}

// Decoration: the movement is named in text directly above this, so the figure carries an empty
// alt rather than repeating it to a screen reader.
export default function MovementArt({ movement }: Props) {
  const file = ART[movement];

  // No movement in the catalog is missing art today; the MovementIcon fallback stays as a guard
  // against a future movement being added to lib/movements.ts before its art ships.
  if (!file) {
    return (
      <span className="flex h-full w-full items-center justify-center">
        <MovementIcon movement={movement} size={84} strokeWidth={1.15} dim />
      </span>
    );
  }

  return (
    <img
      src={`/movements/${file}`}
      alt=""
      loading="lazy"
      decoding="async"
      className="h-full w-full object-contain"
    />
  );
}
