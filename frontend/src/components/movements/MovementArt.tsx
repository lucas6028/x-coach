import MovementIcon from "./MovementIcon";

// The illustration shown on a movement card, keyed by the canonical English movement name.
//
// These are the exercise_library_muse-spark reference's own figures, cut out and cleaned before
// being committed: the sources are scraped stock art -- one carried a stock-agency watermark bar,
// one a third-party wordmark, and their backgrounds ranged from flat white through a faint grid to
// a photograph of a swimming pool. They were knocked out to transparency, trimmed and resized from
// 6.9 MB to ~1 MB for the set. `scripts/prep_movement_art.py` is that pipeline, kept so the eight
// can be regenerated or audited rather than being opaque binaries.
//
// KNOWN GAP, stated rather than hidden: the reference only drew eight exercises and this catalog
// has sixteen. The rest fall back to their own MovementIcon figure drawn large -- the same drawing
// that sits beside the title, so a card without an illustration still shows ITS movement rather
// than a stand-in for movement in general.
export const ART: Record<string, string> = {
  Squat: "squat.png",
  Lunge: "lunge.png",
  Deadlift: "deadlift.png",
  "Push-up": "pushup.png",
  "Overhead Press": "overhead-press.png",
  Row: "row.png",
  "Sit-up": "situp.png",
  "Jumping Jacks": "jumping-jack.png",
};

interface Props {
  /** Canonical English movement name — the key, not the label the reader sees. */
  movement: string;
}

// Decoration: the movement is named in text directly above this, so the figure carries an empty
// alt rather than repeating it to a screen reader.
export default function MovementArt({ movement }: Props) {
  const file = ART[movement];

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
