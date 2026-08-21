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
//
// EXCEPTIONS to the stroke rule: Squat, Lunge, Deadlift, Push-up, Overhead Press, Bicep Curl and
// Sit-up are filled silhouettes (`fill`), traced from supplied pictograms rather than drawn here.
// Deliberate departures asked for by name, not oversights -- do not "fix" them back into strokes.
// They bend the waist-up rule too, and they are visibly heavier marks than their stroked
// neighbours at 13-18px; both are the cost of using the supplied art.
//
// The trace, for whoever adds the next one: threshold the source at gray < 200, take the contour
// TREE (not just the outer one -- these sources have holes, and separate pieces wherever a white
// keyline cuts through), and pull out the one near-perfect disc as the head, which stays a
// `<circle>`. Simplify each contour with approxPolyDP, smooth to cubics, emit them as subpaths of
// one `d`, and fit into this frame. Never ship the source PNG as an `<img>` instead: those files
// are RGB with no alpha, so they render as a white box over the card's gradient, and their violet
// is baked in -- `dim` works only through `currentColor`.

interface Glyph {
  /** Head centre. */
  head: [number, number];
  /** Head radius, when the set's r=2 is wrong for it. Only silhouettes need this: each is traced
   *  from its own source art, so a head that reads as attached to the arms there can come back
   *  meaningfully off 2 here, and forcing it detaches the head. Stroked glyphs all use 2. */
  headR?: number;
  /** SVG path data, stroked at `strokeWidth`. */
  strokes?: string[];
  /** SVG path data, filled with no stroke. The silhouette alternative to `strokes`; a glyph
   *  carries one or the other, plus the head. `strokeWidth` is inert on it. */
  fill?: string;
}

// Keys are the canonical English movement names from lib/movements.ts, verbatim.
export const GLYPHS: Record<string, Glyph> = {
  // ── Lower body ────────────────────────────────────────────────────────────
  // Side view: hips down and back, knee forward over the foot, arms counterweighted forward.
  // Silhouette (see the exception above): 23 anchors, fitted 19.4 units tall with the foot on
  // y=22, which is where the traced head landed at r=2.05 -- near enough to the shared circle.
  Squat: {
    head: [11.9, 4.6],
    fill: "M10.02 6.78C9.55 6.89 9.43 6.61 8.63 7.7C7.83 8.78 5.83 12.14 5.22 13.31C4.61 14.48 4.93 14.36 4.97 14.74C5 15.13 5.18 15.4 5.41 15.64C5.64 15.87 5.47 16.07 6.35 16.15C7.23 16.24 10.22 15.23 10.7 16.15C11.18 17.07 9.41 20.7 9.24 21.67C9.07 22.65 9.06 21.95 9.69 22C10.31 22.05 12.36 22.16 12.97 22C13.59 21.84 13.56 21.35 13.4 21.04C13.23 20.73 11.87 21.06 11.99 20.14C12.11 19.23 13.77 16.53 14.13 15.56C14.48 14.6 14.28 14.68 14.13 14.34C13.97 14 13.92 13.74 13.21 13.52C12.5 13.3 10.09 13.53 9.87 13C9.66 12.48 11.16 10.67 11.89 10.4C12.63 10.12 13.19 11.19 14.29 11.36C15.39 11.53 17.72 11.57 18.52 11.41C19.31 11.25 19.08 10.69 19.06 10.4C19.04 10.11 19.14 9.87 18.4 9.67C17.66 9.47 15.8 9.66 14.64 9.22C13.48 8.78 12.22 7.45 11.45 7.04C10.68 6.63 10.49 6.67 10.02 6.78Z",
  },
  // Side view at the bottom: front shin vertical, trailing knee dropped to the floor and that
  // shin flat along it. Silhouette (see the exception above): one subpath, 23 anchors, fitted
  // 19.4 units tall standing on y=22 like Squat.
  Lunge: {
    head: [13.1, 4.8],
    headR: 2.2,
    fill: "M14.34 7.77C14.05 7.44 13.86 7.43 13.52 7.38C13.18 7.32 12.63 7.35 12.3 7.43C11.98 7.51 11.78 7.65 11.56 7.85C11.35 8.06 11.07 7.61 11.01 8.67C10.95 9.74 11.42 12.51 11.19 14.24C10.97 15.97 10.72 18.16 9.64 19.07C8.55 19.98 5.59 19.46 4.68 19.7C3.76 19.95 4.15 20.22 4.15 20.55C4.14 20.88 3.49 21.49 4.65 21.68C5.81 21.88 9.91 21.84 11.12 21.74C12.32 21.63 11.45 21.81 11.88 21.05C12.31 20.29 12.79 17.91 13.7 17.2C14.61 16.48 16.7 16.04 17.34 16.75C17.99 17.46 17.38 20.57 17.56 21.45C17.73 22.32 18.14 21.92 18.4 22C18.66 22.08 18.93 22 19.14 21.92C19.35 21.84 19.52 21.69 19.64 21.52C19.76 21.36 19.83 22.03 19.85 20.92C19.87 19.8 19.93 16 19.75 14.85C19.56 13.7 19.51 14.18 18.74 14.03C17.97 13.88 15.72 14.73 15.13 13.95C14.54 13.17 15.34 10.39 15.21 9.36C15.07 8.33 14.62 8.1 14.34 7.77Z",
  },
  // Front view at the lift's start: hinged over the bar, arms hanging to it, plates either side.
  // Silhouette (see the exception above), and the most divided one: six subpaths, because the
  // source draws a white keyline where the bar crosses the body, cutting the shins below it and
  // the outer collars off into pieces of their own. The sixth is the gap under the torso, a hole.
  // Wide and short like Push-up, so fitted 20 units across -- but stood on y=22, it has feet.
  Deadlift: {
    head: [12.4, 6],
    fill: "M13.58 16.83C13.38 17.59 13.99 20.58 14.25 21.44C14.51 22.3 14.78 21.91 15.12 21.98C15.46 22.05 16.06 21.97 16.29 21.87C16.52 21.76 16.59 21.64 16.51 21.37C16.44 21.1 16.02 21.01 15.84 20.25C15.66 19.5 15.81 17.42 15.44 16.85C15.06 16.28 13.78 16.06 13.58 16.83ZM10 16.83C9.88 16.06 8.87 16.76 8.56 16.83C8.26 16.89 8.3 16.62 8.16 17.21C8.02 17.79 7.89 19.63 7.71 20.32C7.53 21.01 7.13 21.07 7.06 21.33C6.99 21.59 7.06 21.78 7.31 21.89C7.55 21.99 8.21 22.03 8.54 21.96C8.87 21.88 9.04 22.29 9.28 21.44C9.52 20.59 10.11 17.6 10 16.83ZM21.31 15.73C21.21 15.74 21.23 15.61 21.22 15.8C21.2 15.98 21.12 16.65 21.22 16.83C21.31 17.01 21.67 16.9 21.8 16.87C21.93 16.84 21.97 16.8 22 16.65C22.03 16.5 22.03 16.12 22 15.98C21.97 15.83 21.91 15.79 21.8 15.75C21.68 15.71 21.4 15.72 21.31 15.73ZM2.13 15.8C2 15.84 2.01 15.88 2 16.04C1.99 16.2 2.03 16.62 2.07 16.76C2.1 16.9 2.1 16.86 2.22 16.87C2.34 16.89 2.69 17.03 2.78 16.85C2.88 16.67 2.89 15.95 2.78 15.77C2.68 15.6 2.27 15.75 2.13 15.8ZM3.08 14.12C2.98 14.91 2.95 17.51 3.05 18.33C3.15 19.14 3.33 18.94 3.68 19C4.03 19.05 4.89 19.07 5.16 18.66C5.42 18.26 4.99 16.93 5.27 16.58C5.55 16.23 6.52 16.51 6.86 16.58C7.2 16.65 7.08 17.01 7.33 17.01C7.58 17.01 6.98 16.65 8.36 16.58C9.75 16.51 14.36 16.51 15.64 16.58C16.92 16.65 15.55 17.01 16.06 17.01C16.58 17.01 18.24 16.27 18.73 16.58C19.22 16.89 18.72 18.44 19 18.84C19.28 19.24 20.09 19.06 20.41 18.98C20.73 18.89 20.88 19.2 20.95 18.35C21.01 17.5 21.12 14.65 20.81 13.87C20.5 13.09 19.43 13.29 19.09 13.65C18.75 14 19.07 15.62 18.75 16.02C18.44 16.42 17.68 17.19 17.18 16.06C16.68 14.94 16.12 10.59 15.75 9.28C15.39 7.96 15.3 8.45 14.99 8.18C14.68 7.92 14.38 7.68 13.91 7.69C13.45 7.69 12.81 8.21 12.19 8.2C11.57 8.2 10.75 7.63 10.17 7.67C9.6 7.7 9.09 8.13 8.76 8.43C8.44 8.72 8.55 8.16 8.23 9.44C7.9 10.71 7.31 14.97 6.82 16.06C6.32 17.16 5.56 16.42 5.25 16.02C4.93 15.62 5.18 14.06 4.91 13.65C4.64 13.24 3.94 13.48 3.63 13.56C3.33 13.63 3.17 13.32 3.08 14.12ZM9.7 10.22C9.75 10.2 8.72 14.22 8.56 15.15C8.41 16.08 8.52 15.69 8.79 15.8C9.05 15.9 9.79 16.12 10.13 15.8C10.47 15.47 10.56 14.15 10.8 13.83C11.05 13.5 11.21 13.5 11.61 13.83C12.01 14.15 12.6 15.47 13.2 15.8C13.8 16.12 15.18 16.4 15.21 15.77C15.25 15.15 13.58 12.97 13.42 12.06C13.27 11.14 13.89 9.76 14.27 10.29C14.65 10.82 15.51 14.27 15.71 15.24C15.9 16.2 16.64 15.93 15.44 16.06C14.24 16.2 9.71 16.18 8.52 16.04C7.32 15.91 8.07 16.23 8.27 15.26C8.47 14.29 9.66 10.24 9.7 10.22Z",
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
  // Top of the plank, seen from the side: shoulders over a vertical arm, body sloping down to the
  // toes. Silhouette (see the exception above): 22 anchors. This figure is wide and short, so it
  // is fitted the other way round from Squat -- 20 units across, then centred vertically rather
  // than stood on the ground line, the same placement the stroked version used.
  "Push-up": {
    head: [20.2, 9.4],
    fill: "M19.31 15.96C19.35 15.8 19.25 15.58 19.08 15.46C18.91 15.33 18.43 15.68 18.28 15.2C18.13 14.72 18.14 13.27 18.18 12.58C18.21 11.88 18.46 11.42 18.49 11.04C18.52 10.67 18.48 10.56 18.37 10.33C18.25 10.1 18.02 9.79 17.8 9.64C17.57 9.48 18.4 9.01 17.02 9.38C15.64 9.76 11.95 11.19 9.54 11.91C7.14 12.62 3.81 13.33 2.59 13.67C1.37 14.01 2.31 13.82 2.21 13.94C2.11 14.07 2.01 14.09 2 14.43C1.99 14.76 2.05 15.65 2.15 15.96C2.24 16.27 2.42 16.23 2.57 16.3C2.71 16.36 2.84 16.39 3.01 16.34C3.17 16.29 3.42 16.17 3.55 16C3.69 15.84 2.67 15.69 3.81 15.35C4.94 15.01 8.28 14.47 10.38 13.96C12.49 13.46 15.33 11.94 16.43 12.3C17.54 12.67 16.82 15.47 17 16.15C17.18 16.83 17.22 16.36 17.53 16.4C17.83 16.44 18.55 16.47 18.85 16.4C19.15 16.33 19.27 16.12 19.31 15.96Z",
  },
  // Lockout, bar overhead. Silhouette (see the exception above): two subpaths, the second being
  // the space the arms and the bar close around. The head is an island inside that hole, which is
  // the reason it stays a separate `<circle>` here rather than joining the fill.
  // Full body, not waist up -- the drawing rule above bends for the silhouettes, whose sources
  // come with legs.
  "Overhead Press": {
    head: [12, 7.3],
    headR: 1.51,
    fill: "M6.63 2.64C6.44 2.66 6.34 2.78 6.28 3.05C6.22 3.31 6.44 4.02 6.26 4.23C6.08 4.44 5.41 4.2 5.21 4.29C5.01 4.38 4.88 4.67 5.06 4.79C5.23 4.91 6.06 4.77 6.26 5.01C6.47 5.25 6.2 5.96 6.28 6.23C6.36 6.5 6.54 6.63 6.73 6.64C6.92 6.65 7.3 6.58 7.43 6.31C7.55 6.03 7.33 5.21 7.47 4.99C7.6 4.77 8.2 4.38 8.22 4.97C8.25 5.56 7.31 7.57 7.6 8.54C7.89 9.52 9.55 9.95 9.97 10.81C10.39 11.68 10.28 11.93 10.11 13.71C9.93 15.48 8.99 20.07 8.9 21.46C8.82 22.84 9.33 21.98 9.6 22C9.87 22.02 10.11 22.69 10.51 21.57C10.92 20.46 11.56 15.32 12.05 15.3C12.53 15.28 13.11 20.33 13.43 21.44C13.74 22.54 13.74 21.85 13.93 21.94C14.12 22.03 14.37 22.12 14.57 21.96C14.78 21.81 15.29 22.39 15.18 21.01C15.07 19.63 14.09 15.43 13.91 13.71C13.73 11.99 13.67 11.53 14.09 10.68C14.5 9.82 16.12 9.53 16.4 8.58C16.68 7.63 15.75 5.57 15.78 4.97C15.8 4.37 16.4 4.74 16.53 4.97C16.67 5.2 16.47 6.05 16.59 6.33C16.72 6.61 17.1 6.66 17.29 6.64C17.48 6.62 17.64 6.5 17.72 6.23C17.79 5.96 17.57 5.22 17.74 5.01C17.91 4.8 18.54 5.05 18.75 4.97C18.95 4.89 19.13 4.62 18.96 4.5C18.79 4.38 17.95 4.5 17.74 4.25C17.53 4 17.8 3.3 17.72 3.03C17.63 2.75 17.42 2.61 17.23 2.6C17.04 2.59 16.7 2.68 16.57 2.97C16.45 3.25 18.01 4.09 16.5 4.31C14.98 4.53 9.02 4.54 7.5 4.31C5.99 4.08 7.55 3.19 7.41 2.91C7.26 2.63 6.82 2.62 6.63 2.64ZM9.45 4.97C10.36 4.49 13.7 4.46 14.59 4.95C15.49 5.44 15.1 7.23 14.83 7.92C14.55 8.61 13.62 8.93 12.94 9.11C12.26 9.28 11.38 9.18 10.75 8.97C10.11 8.76 9.35 8.49 9.14 7.82C8.92 7.16 8.54 5.45 9.45 4.97Z",
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
  // Front view, both arms curled: elbows at the ribs, forearms turned up, a dumbbell in each hand.
  // Silhouette (see the exception above): one subpath, 51 anchors, and the anchors are almost all
  // spent on the two dumbbells -- shrink it further and they round off into mittens. Full body,
  // not waist up, for the reason given on Overhead Press.
  "Bicep Curl": {
    head: [12, 4.4],
    headR: 1.78,
    fill: "M6.31 6.32C6.14 6.59 5.93 7.74 5.98 8.06C6.03 8.39 6.42 8.34 6.61 8.27C6.8 8.21 6.89 7.29 7.11 7.68C7.33 8.07 7.61 10.09 7.94 10.59C8.27 11.1 8.75 10.86 9.09 10.69C9.42 10.52 9.81 9.27 9.95 9.58C10.09 9.88 10.13 10.53 9.93 12.53C9.73 14.53 8.78 20.02 8.74 21.6C8.71 23.18 9.44 22.03 9.72 22C10 21.97 10.09 22.78 10.43 21.42C10.77 20.07 11.48 15.14 11.75 13.85C12.03 12.56 11.77 12.42 12.08 13.68C12.38 14.95 13.23 20.06 13.57 21.44C13.91 22.82 13.87 21.89 14.11 21.96C14.35 22.03 14.78 22.04 14.99 21.87C15.2 21.7 15.53 22.5 15.37 20.95C15.22 19.4 14.29 14.46 14.07 12.57C13.85 10.68 13.9 9.9 14.05 9.6C14.2 9.29 14.65 10.55 14.97 10.73C15.29 10.9 15.67 11.16 15.99 10.65C16.31 10.14 16.66 8.06 16.89 7.66C17.12 7.26 17.2 8.21 17.39 8.27C17.58 8.34 17.95 8.35 18.02 8.04C18.09 7.73 17.95 6.69 17.79 6.41C17.63 6.14 17.21 6.25 17.08 6.38C16.95 6.5 17.2 7.09 17.02 7.18C16.84 7.28 16.31 6.9 16.01 6.95C15.7 7 15.36 7.51 15.18 7.47C15.01 7.43 15.11 6.82 14.95 6.72C14.8 6.62 14.33 6.56 14.26 6.87C14.19 7.19 14.4 8.34 14.55 8.62C14.7 8.9 15.01 8.7 15.16 8.58C15.31 8.46 15.44 7.79 15.45 7.89C15.46 7.99 15.43 9.06 15.24 9.18C15.05 9.29 14.5 8.95 14.3 8.6C14.1 8.25 14.16 7.38 14.05 7.07C13.94 6.75 14.23 6.8 13.65 6.72C13.07 6.65 11.2 6.57 10.58 6.63C9.96 6.68 10.1 6.72 9.95 7.07C9.8 7.41 9.86 8.33 9.66 8.68C9.46 9.03 8.95 9.31 8.76 9.18C8.57 9.04 8.54 7.99 8.55 7.89C8.56 7.79 8.66 8.44 8.82 8.56C8.97 8.68 9.34 8.89 9.49 8.6C9.64 8.31 9.79 7.15 9.72 6.84C9.65 6.52 9.21 6.62 9.07 6.72C8.92 6.83 9.02 7.43 8.84 7.47C8.66 7.51 8.27 6.99 7.99 6.95C7.71 6.91 7.32 7.31 7.15 7.22C6.98 7.13 7.12 6.58 6.98 6.43C6.84 6.28 6.47 6.05 6.31 6.32Z",
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
  // Halfway up: hips on the floor, torso raised, hands behind the head, knees bent. Silhouette
  // (see the exception above): 30 anchors, the most in the set -- the arm looping round the head
  // and the crossed shins are what the pose is, and simplifying past this smooths them away.
  // Fitted 20 units across with the hips on y=20.8, the ground line the stroked version drew.
  "Sit-up": {
    head: [17.9, 9.1],
    headR: 2.32,
    fill: "M21.93 11.44C21.71 10.99 20.74 9.99 20.32 9.95C19.9 9.91 19.48 10.86 19.41 11.19C19.34 11.51 20.03 11.73 19.9 11.91C19.77 12.09 19.11 12.23 18.64 12.26C18.16 12.29 17.52 12.22 17.05 12.1C16.59 11.97 16.21 11.8 15.84 11.54C15.47 11.27 14.89 10.73 14.84 10.51C14.78 10.28 15.43 10.48 15.51 10.18C15.59 9.88 15.64 8.91 15.3 8.69C14.96 8.47 13.85 8.75 13.46 8.87C13.07 9 13.04 9.21 12.95 9.41C12.85 9.62 12.67 9.45 12.9 10.11C13.13 10.77 14.4 12.34 14.35 13.38C14.29 14.42 13.41 16.34 12.55 16.37C11.68 16.39 9.88 14.06 9.16 13.54C8.45 13.02 8.58 13.27 8.28 13.24C7.98 13.21 7.67 13.25 7.37 13.38C7.06 13.51 7.32 13.05 6.46 14.03C5.59 15.02 2.92 18.25 2.19 19.28C1.46 20.31 1.96 19.97 2.07 20.22C2.18 20.47 2.44 20.71 2.86 20.78C3.28 20.84 3.76 21.28 4.59 20.59C5.42 19.9 6.67 16.65 7.86 16.62C9.05 16.6 10.89 19.75 11.73 20.45C12.57 21.15 12.4 20.78 12.9 20.8C13.4 20.82 14.1 20.93 14.72 20.59C15.33 20.25 15.9 19.74 16.59 18.75C17.27 17.75 17.98 15.65 18.83 14.64C19.67 13.62 21.13 13.19 21.65 12.66C22.17 12.12 22.15 11.89 21.93 11.44Z",
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
      <circle
        cx={glyph.head[0]}
        cy={glyph.head[1]}
        r={glyph.headR ?? 2}
        fill="currentColor"
        stroke="none"
      />
      {/* evenodd, not the default nonzero: a silhouette is one path of several subpaths, and the
          nested ones are holes -- the space the press's arms close around, the gap under the
          deadlift's torso. Harmless on the single-subpath glyphs. */}
      {glyph.fill && (
        <path d={glyph.fill} fill="currentColor" fillRule="evenodd" stroke="none" />
      )}
      {glyph.strokes?.map((d) => (
        <path key={d} d={d} />
      ))}
    </svg>
  );
}
