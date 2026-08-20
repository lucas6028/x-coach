// The small glyph beside a movement's name on its card, as in the exercise_library_muse-spark
// reference. Each one is drawn here, as a stick figure IN THAT MOVEMENT'S POSITION.
//
// Not a general icon set: the stock libraries have no shoulder bridge, no band pull apart and no
// arm V-W, so keying to them means borrowing something adjacent -- a bed for the bridge, a wind
// glyph for a torso twist -- and an icon that shows the wrong thing is worse than none.
//
// DRAWING RULES, so the sixteen read as one set:
//  * 24x24 frame, ground at y=20-22, figure roughly 18 units tall.
//  * Filled head, everything else a 1.7-wide round-capped stroke. At 18px on screen a limb is
//    about two pixels wide, so each figure gets the fewest strokes that still name the movement:
//    one arm and one leg for anything seen from the side, both when the two differ (a lunge, a
//    high knee) or when the movement is the symmetry (a jumping jack).
//  * The arm movements are drawn from the waist up. Legs that do nothing cost a third of the
//    frame's height, and shrink the part that carries the meaning.
//  * Equipment is drawn: a bar with plates, a dumbbell, the band's slack curve. It is often the
//    fastest thing to recognise at this size.

interface Glyph {
  /** Head centre. */
  head: [number, number];
  /** SVG path data, stroked. */
  strokes: string[];
}

// Keys are the canonical English movement names from lib/movements.ts, verbatim.
export const GLYPHS: Record<string, Glyph> = {
  // ── Lower body ────────────────────────────────────────────────────────────
  // Side view: hips down and back, knee forward over the foot, arms counterweighted forward.
  Squat: {
    head: [9.5, 4.5],
    strokes: ["M9.5 6.6 L8.2 13", "M9.3 8.2 L14 8.6 L18.5 8.6", "M8.2 13 L13.5 16.2 L10.5 22"],
  },
  // Split stance: front shin vertical, trailing knee dropped behind the hips.
  Lunge: {
    head: [9, 4],
    strokes: [
      "M9 6.1 L9 12.6",
      "M9 8 L11.6 10.6",
      "M9 12.6 L15 16.6 L15 22",
      "M9 12.6 L4.2 17 L5.8 22",
    ],
  },
  // Hinged over the bar, arms hanging straight to it.
  Deadlift: {
    head: [14.8, 5],
    strokes: [
      "M13.2 6.4 L6.6 10.2",
      "M12.4 7.6 L11.8 13.4",
      "M6.6 10.2 L8.8 16.2 L7.6 22",
      "M5 14.2 L18.4 14.2",
      "M6.4 12.4 L6.4 16",
      "M17 12.4 L17 16",
    ],
  },
  // Front view: one leg carried out to the side, the other under the hip.
  "Leg Abduction": {
    head: [11.5, 4],
    strokes: [
      "M11.5 6.1 L11.5 13",
      "M11.5 7.6 L8.6 11",
      "M11.5 7.6 L14.2 10.6",
      "M11.5 13 L10.6 17.4 L10.2 22",
      "M11.5 13 L16.6 16.8 L20.6 20.4",
    ],
  },
  // Supine, hips driven up: shoulders and feet down, thigh level, shin vertical.
  "Shoulder Bridge": {
    head: [3.6, 18.4],
    strokes: [
      "M5.6 18.2 L12.6 13 L18 13 L18.4 20",
      "M2.5 20.8 L21.5 20.8",
    ],
  },

  // ── Upper body ────────────────────────────────────────────────────────────
  // Top of the plank: one line from shoulders to ankles, arm vertical to the floor.
  "Push-up": {
    head: [19.4, 8.4],
    strokes: [
      "M17.6 9.4 L11 12 L4 15.4",
      "M17.6 9.4 L17.6 17.2",
      "M2.5 17.8 L21.5 17.8",
    ],
  },
  // Lockout, bar overhead.
  "Overhead Press": {
    head: [12, 11],
    strokes: [
      "M12 13 L12 18.6",
      "M12 13.2 L8.4 8.6",
      "M12 13.2 L15.6 8.6",
      "M4.6 7.4 L19.4 7.4",
      "M6.4 5.2 L6.4 9.6",
      "M17.6 5.2 L17.6 9.6",
    ],
  },
  // Bent over, one arm hanging with a dumbbell.
  Row: {
    head: [17, 6],
    strokes: [
      "M15.5 7.4 L8 10.6",
      "M14.2 8.8 L14.2 13.6",
      "M11.6 13.6 L16.8 13.6",
      "M8 10.6 L9.6 16.4 L8.4 22",
    ],
  },
  // Waist up: elbows at the ribs, forearms turned up, a dumbbell in each hand.
  "Bicep Curl": {
    head: [12, 5],
    strokes: [
      "M12 7.1 L12 17.5",
      "M10.8 8.6 L8.4 12.4 L9.6 8.2",
      "M13.2 8.6 L15.6 12.4 L14.4 8.2",
      "M7.6 7.4 L11.6 7.4",
      "M12.4 7.4 L16.4 7.4",
    ],
  },
  // Waist up: arms straight out at shoulder height, the band slack across the chest.
  "Band Pull Apart": {
    head: [12, 5],
    strokes: [
      "M12 7.1 L12 17.5",
      "M3.6 11 L12 9.6 L20.4 11",
      "M3.6 11 Q12 15.4 20.4 11",
    ],
  },
  // Waist up: arms raised out and up on a diagonal.
  "Arm Abduction": {
    head: [12, 5],
    strokes: ["M12 7.1 L12 17.5", "M4.4 5.6 L12 9.8 L19.6 5.6"],
  },
  // Waist up: the W of the V-W drill — upper arms out, elbows bent, forearms up.
  "Arm VW": {
    head: [12, 5],
    strokes: ["M12 7.1 L12 17.5", "M5.6 5 L8.2 11 L12 9.6 L15.8 11 L18.4 5"],
  },

  // ── Core ──────────────────────────────────────────────────────────────────
  // Halfway up: hips on the floor, torso steep, knees bent.
  "Sit-up": {
    head: [14.2, 7],
    strokes: [
      "M13.2 8.7 L6 18.2 L13 15 L17.2 19.6",
      "M2.5 20.8 L21.5 20.8",
    ],
  },
  // Standing rotation: both arms carried across to one side, hips square.
  "Torso Twist": {
    head: [11.5, 4.5],
    strokes: [
      "M11.5 6.6 L11.5 14",
      "M11.5 8.2 L16 10.2 L19.2 9",
      "M11.5 14 L9.8 20.8",
      "M11.5 14 L13.4 20.8",
    ],
  },

  // ── Full body ─────────────────────────────────────────────────────────────
  // Mid-jump: arms overhead and out, feet wide.
  "Jumping Jacks": {
    head: [12, 4],
    strokes: [
      "M12 6.1 L12 13",
      "M4.6 3.6 L12 8 L19.4 3.6",
      "M12 13 L7 21",
      "M12 13 L17 21",
    ],
  },
  // One knee above hip height, opposite arm up.
  "High Knee": {
    head: [10.6, 4],
    strokes: [
      "M10.6 6.1 L10.6 12.6",
      "M10.6 7.6 L14.6 5.4 L15.8 2.4",
      "M10.6 7.6 L7.2 10.6",
      "M10.6 12.6 L9.8 17 L9.4 22",
      "M10.6 12.6 L15.8 10.4 L19 14",
    ],
  },
};

interface Props {
  /** Canonical English movement name — the key, not the label the reader sees. */
  movement: string;
  /** Muted for movements that are listed but not yet analyzable. */
  dim?: boolean;
  size?: number;
  /** In viewBox units. Scale it DOWN as `size` goes up: at the 18px default 1.7 is a hairline,
   *  but blown up to fill a card's tile the same figure comes out drawn in fat sausages. */
  strokeWidth?: number;
}

// Decorative: the name it sits beside says the same thing, so it is hidden from assistive tech.
export default function MovementIcon({
  movement,
  dim,
  size = 18,
  strokeWidth = 1.7,
}: Props) {
  const glyph = GLYPHS[movement];
  if (!glyph) return null;

  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={strokeWidth}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
      className={`shrink-0 ${dim ? "text-faint" : "text-primary"}`}
    >
      <circle cx={glyph.head[0]} cy={glyph.head[1]} r={2} fill="currentColor" stroke="none" />
      {glyph.strokes.map((d) => (
        <path key={d} d={d} />
      ))}
    </svg>
  );
}
