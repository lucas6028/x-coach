// The small glyph beside a movement's name on its card, as in the exercise_library_muse-spark
// reference. Each one is drawn here, as a stick figure IN THAT MOVEMENT'S POSITION.
//
// Not a general icon set: the stock libraries have no shoulder bridge, no band pull apart and no
// arm V-W, so keying to them means borrowing something adjacent -- a bed for the bridge, a wind
// glyph for a torso twist -- and an icon that shows the wrong thing is worse than none.
//
// TWO KINDS OF GLYPH live here, and the set reads as two visual languages until that stops:
//
//  * FILLED SILHOUETTES (`fill`) -- fifteen of the sixteen, traced from supplied pictograms rather
//    than drawn here. Asked for by name, one movement at a time; not oversights, so do not "fix"
//    them back into strokes. They are heavier marks than a stroked figure at 13-18px, and they
//    ignore the waist-up rule below, their sources having come with legs.
//  * A STROKED STICK FIGURE (`strokes`) -- Row, alone now, still waiting on silhouette art. The
//    drawing rules are its. Keep the `strokes` branch until Row is converted; after that both it
//    and these rules are dead, and the whole file collapses to one shape.
//
// Both kinds share the 24x24 frame and the `<circle>` head.
//
// DRAWING RULES for the last stroked figure:
//  * 24x24 frame, ground at y=20-22, figure roughly 18 units tall.
//  * Filled head, everything else a 1.7-wide round-capped stroke. At 18px on screen a limb is
//    about two pixels wide, so a figure gets the fewest strokes that still name the movement:
//    one arm and one leg for anything seen from the side.
//  * The arm movements are drawn from the waist up. Legs that do nothing cost a third of the
//    frame's height, and shrink the part that carries the meaning.
//  * Equipment is drawn -- what is left of it is Row's dumbbell. It is often the fastest thing to
//    recognise at this size.
//
// THE TRACE, for whoever adds the next silhouette: threshold the source at gray < 200, take the
// contour TREE (not just the outer one -- these sources have holes, and separate pieces wherever
// a white keyline cuts through), and pull out the one near-perfect disc as the head, which stays
// the `<circle>`. Discard contours under ~20 square pixels first -- antialiasing leaves specks
// that a trace reads as holes. Simplify each contour with approxPolyDP, smooth to cubics, emit
// them as subpaths of one `d`, and fit into this frame. Never ship the source PNG as an `<img>`
// instead: those files are RGB with no alpha, so they render as a white box over the card's
// gradient, and their violet is baked in -- `dim` works only through `currentColor`.

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
  // Front view: hands on the hips, one leg carried out to the side, the other under the hip.
  // Silhouette (see the exception above): one subpath, 41 anchors. The triangles between each arm
  // and the torso look enclosed but are not -- hand and hip stop just short of meeting in the
  // source, so they trace as open notches, not holes.
  "Leg Abduction": {
    head: [15.1, 4.1],
    headR: 1.49,
    fill: "M13.88 5.86C13.33 5.92 13.44 5.86 13.02 6.22C12.6 6.57 11.69 7.6 11.38 7.97C11.07 8.34 11.19 8.26 11.16 8.43C11.13 8.6 11.03 8.6 11.22 8.99C11.41 9.38 12.03 10.45 12.3 10.77C12.57 11.1 12.72 10.96 12.86 10.95C13 10.94 13.1 10.83 13.16 10.73C13.22 10.64 13.35 10.71 13.24 10.37C13.13 10.03 12.49 9.07 12.48 8.67C12.47 8.27 13.06 7.65 13.2 7.97C13.34 8.29 13.36 10.06 13.32 10.59C13.28 11.13 13.62 10.84 12.96 11.19C12.3 11.54 10.49 12.16 9.36 12.67C8.24 13.18 6.88 14.02 6.21 14.25C5.53 14.48 5.51 13.99 5.31 14.05C5.1 14.11 4.97 14.35 4.99 14.61C5 14.87 5.22 15.4 5.39 15.61C5.55 15.81 5.65 15.91 5.99 15.83C6.33 15.74 6.8 15.35 7.42 15.09C8.05 14.82 8.64 14.59 9.74 14.25C10.85 13.91 13.27 13.22 14.06 13.03C14.84 12.84 14.36 13.04 14.46 13.09C14.55 13.14 14.59 11.92 14.62 13.33C14.64 14.74 14.55 20.1 14.6 21.54C14.65 22.98 14.7 21.9 14.92 21.98C15.14 22.06 15.68 22.07 15.92 22C16.15 21.93 16.33 21.74 16.32 21.54C16.31 21.34 15.79 22.18 15.86 20.78C15.92 19.38 16.51 15.27 16.7 13.15C16.88 11.02 16.79 8.78 16.95 8.03C17.12 7.29 17.68 8.26 17.67 8.65C17.66 9.04 17 10.02 16.89 10.37C16.79 10.73 16.94 10.69 17.01 10.79C17.08 10.89 17.17 10.98 17.31 10.97C17.46 10.96 17.62 11.05 17.87 10.75C18.13 10.45 18.66 9.52 18.85 9.15C19.04 8.79 19.02 8.75 19.01 8.55C19 8.36 19.15 8.41 18.79 7.99C18.44 7.57 17.31 6.39 16.89 6.04C16.48 5.68 16.82 5.89 16.32 5.86C15.81 5.83 14.43 5.8 13.88 5.86Z",
  },
  // Supine, hips driven up: head and shoulders on the floor at one end, feet at the other, the
  // body a ramp between them. Silhouette (see the exception above): two subpaths, the smaller
  // being the far shin, which the source's white keyline cuts off from the rest. At 2.7:1 this is
  // the flattest figure in the set, so like Push-up it is fitted across -- 20 units -- and centred
  // vertically; standing it on a ground line would leave two thirds of the frame empty above it.
  "Shoulder Bridge": {
    head: [3.8, 13.9],
    headR: 1.79,
    fill: "M16.99 11.31C17.04 12.01 17.79 14.59 18.03 15.32C18.26 16.06 18.24 15.63 18.42 15.7C18.6 15.77 19.21 16.48 19.1 15.72C18.99 14.96 18.1 11.86 17.74 11.12C17.39 10.39 16.94 10.61 16.99 11.31ZM22 15.31C22.01 15.12 21.96 14.82 21.76 14.63C21.55 14.44 21.14 15.07 20.78 14.16C20.41 13.24 19.85 10.06 19.59 9.13C19.33 8.2 19.37 8.7 19.21 8.56C19.05 8.42 19.5 8.16 18.63 8.28C17.76 8.4 15.74 8.77 13.98 9.28C12.22 9.79 9.2 10.93 8.06 11.35C6.93 11.77 7.44 11.64 7.18 11.82C6.92 12 6.66 12.25 6.48 12.44C6.3 12.64 6.22 12.75 6.11 12.99C6 13.23 5.83 13.56 5.82 13.89C5.81 14.23 5.93 14.72 6.05 14.98C6.17 15.25 6.38 15.37 6.54 15.49C6.7 15.61 5.7 15.66 7.01 15.7C8.32 15.74 13.1 15.75 14.37 15.72C15.65 15.69 14.6 15.61 14.66 15.51C14.72 15.41 14.79 15.26 14.73 15.14C14.67 15.01 15.51 14.93 14.32 14.74C13.13 14.55 8.64 14.12 7.59 13.97C6.55 13.82 7.66 13.83 8.05 13.84C8.43 13.85 9.38 14.15 9.93 14.02C10.48 13.9 10.67 13.31 11.34 13.08C12.02 12.86 13.44 12.8 13.98 12.67C14.51 12.54 14.38 12.47 14.56 12.31C14.74 12.15 14.5 11.97 15.05 11.73C15.6 11.48 17.16 10.19 17.86 10.82C18.55 11.45 18.95 14.7 19.23 15.51C19.52 16.33 19.16 15.68 19.57 15.72C19.98 15.75 21.31 15.79 21.72 15.72C22.12 15.65 21.99 15.49 22 15.31Z",
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
  // Front view at the end of the pull: arms straight out at shoulder height, the band taut
  // between the two handles. Silhouette (see the exception above): two subpaths, the second being
  // the sliver of white between the band and the arms above it -- a hole, and the only thing
  // separating band from body, since they meet at the handles. Full body, not waist up, for the
  // reason given on Overhead Press.
  //
  // The source carries three 3-pixel antialiasing specks that a contour trace reads as holes.
  // Anything under ~20 square pixels is noise; drop it before building the path.
  "Band Pull Apart": {
    head: [12, 4.4],
    headR: 1.76,
    fill: "M4.27 6.8C4.08 6.93 3.86 7.29 3.78 7.46C3.69 7.64 3.72 7.69 3.76 7.83C3.8 7.98 3.88 8.23 3.99 8.36C4.11 8.49 4.22 8.62 4.44 8.63C4.66 8.64 4.81 8.42 5.34 8.42C5.86 8.41 6.85 8.58 7.57 8.59C8.3 8.6 9.29 8.17 9.69 8.48C10.1 8.78 9.97 9.81 10.02 10.4C10.08 10.99 10.25 10.35 10.01 12.02C9.76 13.68 8.91 18.87 8.55 20.4C8.18 21.94 7.93 20.97 7.83 21.2C7.72 21.44 7.85 21.67 7.92 21.81C8 21.94 8.09 21.98 8.29 22C8.5 22.02 8.87 22 9.15 21.9C9.43 21.8 9.52 22.73 9.95 21.4C10.38 20.06 11.39 15.18 11.74 13.91C12.08 12.63 11.92 13.74 12.01 13.75C12.1 13.76 11.95 12.68 12.28 13.94C12.62 15.21 13.61 20 14.01 21.32C14.42 22.64 14.47 21.75 14.71 21.86C14.96 21.98 15.26 22 15.47 22C15.69 22 15.9 21.98 16.02 21.86C16.14 21.74 16.29 21.54 16.19 21.28C16.1 21.02 15.8 21.84 15.43 20.33C15.07 18.81 14.26 13.84 14.01 12.17C13.77 10.51 13.94 10.92 13.98 10.32C14.01 9.73 13.43 8.91 14.21 8.59C14.99 8.28 17.85 8.41 18.66 8.42C19.48 8.42 18.91 8.6 19.07 8.63C19.24 8.66 19.48 8.67 19.64 8.61C19.8 8.56 19.95 8.44 20.05 8.3C20.15 8.17 20.27 8.02 20.24 7.8C20.21 7.57 20.05 7.14 19.89 6.96C19.73 6.77 19.47 6.71 19.31 6.69C19.15 6.66 19.04 6.74 18.94 6.82C18.84 6.91 19.08 7.14 18.7 7.19C18.32 7.25 17.67 7.28 16.66 7.15C15.65 7.03 13.71 6.56 12.67 6.45C11.63 6.35 11.15 6.43 10.41 6.53C9.68 6.63 9.11 6.93 8.25 7.04C7.4 7.15 5.86 7.24 5.3 7.19C4.74 7.14 5.06 6.79 4.89 6.73C4.72 6.66 4.45 6.68 4.27 6.8ZM4.81 8.28C4.78 8.28 5.01 8.03 5.04 7.87C5.08 7.72 4.98 7.41 5 7.37C5.03 7.33 2.88 7.59 5.18 7.64C7.48 7.69 16.5 7.69 18.8 7.64C21.1 7.59 18.97 7.33 19 7.37C19.02 7.41 18.92 7.74 18.96 7.89C18.99 8.05 19.22 8.29 19.19 8.28C19.16 8.28 21.13 7.94 18.8 7.87C16.47 7.81 7.53 7.81 5.2 7.87C2.87 7.94 4.84 8.28 4.81 8.28Z",
  },
  // Front view at the top: both arms straight out at shoulder height. Silhouette (see the
  // exception above): one subpath, 22 anchors -- the fewest of any silhouette, this being little
  // more than a cross. Full body, not waist up, for the reason given on Overhead Press.
  "Arm Abduction": {
    head: [12, 4.4],
    headR: 1.77,
    fill: "M4.37 6.75C2.68 6.91 4.05 7.25 4.08 7.47C4.11 7.68 3.6 7.89 4.55 8.04C5.5 8.18 8.89 6.12 9.78 8.34C10.67 10.56 9.84 19.12 9.9 21.36C9.96 23.59 10.02 21.67 10.14 21.77C10.27 21.88 10.45 21.99 10.65 22C10.86 22.01 11.18 21.95 11.36 21.85C11.53 21.74 11.63 22.75 11.7 21.37C11.77 20 11.7 14.9 11.77 13.59C11.84 12.27 12.03 12.19 12.11 13.47C12.2 14.76 12.21 19.93 12.27 21.3C12.32 22.67 12.36 21.59 12.45 21.7C12.55 21.8 12.61 21.91 12.81 21.94C13.02 21.97 13.49 21.97 13.71 21.87C13.92 21.76 14.02 23.55 14.1 21.3C14.19 19.05 13.32 10.59 14.2 8.38C15.08 6.17 18.44 8.19 19.39 8.06C20.34 7.93 19.81 7.79 19.88 7.6C19.95 7.42 19.89 7.1 19.81 6.94C19.72 6.78 20.3 6.7 19.37 6.64C18.44 6.57 16.7 6.5 14.2 6.52C11.7 6.54 6.05 6.59 4.37 6.75Z",
  },
  // The W of the V-W drill: upper arms out, elbows bent, forearms up. Silhouette (see the
  // exception above): one subpath, 34 anchors, most of them in the two bent arms -- the bend is
  // the whole difference between this and Arm Abduction. Full body, not waist up, for the reason
  // given on Overhead Press.
  "Arm VW": {
    head: [12, 4.4],
    headR: 1.85,
    fill: "M5.8 3.85C5.66 3.9 5.52 4.01 5.45 4.13C5.38 4.26 5.2 3.78 5.37 4.59C5.54 5.41 6.15 8.22 6.46 9.01C6.77 9.79 6.65 9.35 7.23 9.29C7.8 9.23 9.43 8.1 9.88 8.66C10.34 9.22 10 10.52 9.97 12.65C9.93 14.78 9.65 19.92 9.66 21.46C9.67 22.99 9.88 21.79 10.03 21.88C10.17 21.97 10.37 22.01 10.53 22C10.69 21.99 10.84 21.95 10.97 21.84C11.1 21.73 11.18 22.63 11.32 21.34C11.45 20.04 11.68 15.31 11.8 14.08C11.92 12.86 11.9 12.77 12.04 13.98C12.18 15.19 12.46 20.02 12.62 21.34C12.79 22.65 12.82 21.78 13.01 21.88C13.2 21.98 13.58 22 13.77 21.96C13.97 21.92 14.08 21.78 14.18 21.66C14.27 21.53 14.36 22.65 14.34 21.21C14.31 19.78 14.05 15.12 14.01 13.04C13.97 10.95 14.06 9.44 14.1 8.7C14.13 7.97 13.77 8.51 14.22 8.6C14.66 8.7 16.21 9.21 16.75 9.29C17.29 9.37 17.16 9.8 17.46 9.09C17.75 8.38 18.34 5.85 18.53 5.04C18.72 4.23 18.66 4.42 18.59 4.21C18.51 4.01 18.26 3.83 18.06 3.81C17.87 3.79 17.71 3.46 17.44 4.09C17.16 4.73 16.96 7.18 16.41 7.62C15.86 8.06 15.22 6.88 14.14 6.73C13.05 6.58 11 6.58 9.9 6.73C8.81 6.88 8.13 8.03 7.59 7.62C7.04 7.2 6.86 4.86 6.64 4.23C6.42 3.6 6.4 3.91 6.26 3.85C6.12 3.79 5.93 3.8 5.8 3.85Z",
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
  // SEATED rotation, knees bent and arms folded across -- the stroked version showed a STANDING
  // twist, so this changes what the icon depicts, not just how it is drawn. The supplied art is
  // seated; the movement is the same rotation either way.
  // Silhouette (see the exception above): four subpaths. The source's white keylines cut the far
  // leg and the upper body off from the hips, and leave the sliver between the folded forearms and
  // the torso as a hole. Wide and short, so fitted 20 units across, seated on y=22.
  "Torso Twist": {
    head: [8, 8.3],
    headR: 2.14,
    fill: "M6.16 16.71C6.04 17.16 6.5 19.06 6.71 19.72C6.91 20.38 7.14 20.42 7.4 20.66C7.67 20.89 7.86 21.03 8.29 21.13C8.72 21.23 9.55 21.26 10 21.24C10.46 21.23 10.24 21.57 11.02 21.05C11.81 20.52 13.56 17.95 14.71 18.1C15.86 18.24 17.29 21.24 17.92 21.89C18.55 22.54 18.13 22.04 18.49 22C18.84 21.96 19.7 21.79 20.05 21.65C20.39 21.51 20.47 21.31 20.55 21.15C20.63 20.99 20.58 20.84 20.52 20.7C20.47 20.55 20.5 20.41 20.22 20.29C19.94 20.16 19.59 20.68 18.85 19.94C18.12 19.19 16.48 16.64 15.8 15.82C15.11 14.99 15.14 15.09 14.75 14.99C14.37 14.89 14.17 14.94 13.5 15.21C12.83 15.49 11.22 16.51 10.74 16.64C10.26 16.77 10.99 16.02 10.63 15.99C10.27 15.96 9.1 16.3 8.57 16.47C8.04 16.64 7.85 16.99 7.44 17.03C7.04 17.07 6.29 16.26 6.16 16.71ZM15.15 14.93C15.13 15.14 15.42 15 16.08 15.82C16.73 16.64 18.43 19.17 19.07 19.85C19.71 20.54 19.69 19.86 19.92 19.92C20.14 19.97 20.28 20.04 20.42 20.18C20.56 20.32 20.71 20.56 20.76 20.74C20.81 20.93 20.54 21.26 20.72 21.28C20.9 21.31 21.63 21.03 21.85 20.92C22.06 20.8 22 20.77 22 20.61C22 20.45 21.93 20.12 21.85 19.96C21.76 19.81 21.71 19.76 21.48 19.68C21.25 19.6 21.17 20.29 20.46 19.48C19.75 18.68 17.92 15.68 17.21 14.86C16.49 14.04 16.51 14.55 16.16 14.56C15.82 14.57 15.16 14.72 15.15 14.93ZM13.43 14.17C13.21 13.6 12.38 12.01 12 11.44C11.62 10.86 11.46 10.84 11.18 10.7C10.89 10.56 11.15 10.55 10.26 10.59C9.38 10.63 6.71 10.83 5.86 10.92C5.01 11.01 5.37 11.02 5.17 11.13C4.96 11.24 5.12 10.99 4.62 11.57C4.13 12.14 2.61 13.96 2.17 14.58C1.74 15.21 1.99 15.12 2 15.32C2.01 15.52 2.11 15.65 2.22 15.77C2.33 15.9 1.78 15.88 2.65 16.06C3.53 16.23 6.48 16.8 7.47 16.84C8.45 16.87 7.67 16.53 8.57 16.27C9.47 16.01 12.08 15.51 12.87 15.28C13.66 15.04 13.23 15.05 13.32 14.86C13.42 14.68 13.65 14.74 13.43 14.17ZM11.57 13.8C11.11 13.97 9.05 14.72 8.29 14.86C7.54 15.01 7.37 14.62 7.03 14.67C6.7 14.71 6.81 15.14 6.3 15.12C5.78 15.11 4.22 14.67 3.93 14.58C3.64 14.49 4.26 14.78 4.54 14.58C4.82 14.39 5.38 13.37 5.6 13.41C5.82 13.45 5.75 14.59 5.84 14.84C5.93 15.09 5.98 14.99 6.16 14.93C6.35 14.87 6.61 14.54 6.97 14.49C7.32 14.45 7.7 14.75 8.29 14.67C8.88 14.58 10.14 14.26 10.5 13.97C10.87 13.68 10.4 12.96 10.48 12.93C10.56 12.91 10.82 13.68 11 13.82C11.18 13.97 12.02 13.63 11.57 13.8Z",
  },

  // ── Full body ─────────────────────────────────────────────────────────────
  // Mid-jump: arms overhead and out, feet wide. Silhouette (see the exception above): one subpath,
  // 41 anchors. The only symmetric figure in the set, so both arms and both legs are drawn --
  // the symmetry IS the movement, the same reason the stroked version drew all four.
  "Jumping Jacks": {
    head: [12, 5.2],
    headR: 1.89,
    fill: "M6.72 2.72C6.58 2.83 6.37 3.01 6.35 3.3C6.33 3.58 6.37 3.88 6.58 4.42C6.8 4.96 7.14 5.77 7.63 6.55C8.12 7.34 9.17 8.22 9.55 9.15C9.93 10.08 9.86 11.47 9.9 12.12C9.94 12.76 10.28 11.74 9.8 13.03C9.32 14.31 7.58 18.51 6.99 19.81C6.4 21.11 6.35 20.56 6.23 20.86C6.11 21.16 6.2 21.43 6.27 21.61C6.34 21.8 6.52 21.9 6.66 21.96C6.8 22.03 6.85 22.07 7.09 22C7.32 21.93 7.37 22.65 8.08 21.52C8.78 20.38 10.72 16.33 11.33 15.2C11.94 14.06 11.6 14.79 11.72 14.69C11.84 14.59 11.9 14.5 12.07 14.6C12.24 14.7 12.09 14.15 12.73 15.29C13.36 16.44 15.24 20.35 15.89 21.46C16.54 22.57 16.35 21.87 16.62 21.94C16.89 22.01 17.3 21.97 17.49 21.88C17.69 21.8 17.76 21.58 17.8 21.42C17.85 21.25 17.91 21.15 17.79 20.9C17.66 20.64 17.6 21.06 17.05 19.89C16.5 18.71 14.96 15.06 14.47 13.86C13.99 12.66 14.2 13.09 14.14 12.68C14.08 12.27 14.07 11.98 14.12 11.4C14.17 10.82 13.99 10.13 14.43 9.21C14.87 8.29 16.23 6.77 16.76 5.89C17.28 5.02 17.44 4.46 17.57 3.98C17.7 3.49 17.65 3.2 17.55 2.97C17.46 2.74 17.19 2.64 17.01 2.6C16.83 2.56 16.72 2.3 16.45 2.72C16.17 3.13 15.74 4.41 15.36 5.08C14.98 5.75 14.52 6.35 14.16 6.75C13.8 7.15 13.58 7.33 13.19 7.48C12.8 7.64 12.22 7.67 11.84 7.68C11.45 7.68 11.19 7.63 10.9 7.52C10.62 7.41 10.48 7.41 10.11 7.02C9.74 6.63 9.1 5.87 8.7 5.18C8.3 4.49 7.96 3.3 7.71 2.87C7.46 2.44 7.35 2.63 7.18 2.6C7.02 2.57 6.86 2.6 6.72 2.72Z",
  },
  // Mid-stride, seen from the side: one knee driven above hip height, the opposite arm forward.
  // Silhouette (see the exception above): one subpath, 45 anchors. At 9 units across it is the
  // narrowest figure in the set, which is the pose, not a fitting mistake -- everything is stacked
  // front to back, so there is nothing to fill the frame's width with.
  "High Knee": {
    head: [12.5, 4.2],
    headR: 1.63,
    fill: "M13.19 5.96C12.94 5.94 12.66 5.99 12.43 6.09C12.21 6.2 12.06 6.2 11.83 6.59C11.6 6.97 11.44 8.34 11.04 8.42C10.63 8.5 9.75 7.28 9.39 7.06C9.03 6.83 9.03 6.99 8.88 7.06C8.74 7.13 8.55 7.32 8.51 7.49C8.46 7.66 8.28 7.69 8.64 8.08C9 8.47 10.27 9.49 10.68 9.82C11.09 10.14 10.92 10.02 11.09 10.02C11.26 10.02 11.6 9.71 11.7 9.82C11.8 9.92 12.2 10.6 11.7 10.65C11.19 10.69 9.27 10.12 8.66 10.08C8.05 10.04 8.17 10.25 8.03 10.4C7.89 10.55 7.77 10.23 7.83 10.99C7.88 11.75 8.38 14.21 8.34 14.97C8.29 15.74 7.68 15.41 7.54 15.58C7.4 15.74 7.47 15.85 7.5 15.97C7.54 16.09 7.62 16.23 7.73 16.3C7.84 16.36 7.85 16.41 8.15 16.37C8.45 16.33 9.23 16.18 9.53 16.07C9.82 15.96 9.9 16.38 9.92 15.73C9.94 15.08 9.35 12.64 9.64 12.18C9.93 11.71 11.36 12.23 11.64 12.93C11.92 13.63 11.24 15.11 11.34 16.39C11.43 17.67 12.21 19.85 12.21 20.62C12.2 21.4 11.49 20.88 11.3 21.04C11.12 21.19 11.1 21.41 11.09 21.55C11.09 21.69 11.17 21.79 11.26 21.87C11.36 21.94 11.3 21.98 11.66 22C12.02 22.02 13.06 22.02 13.42 21.98C13.77 21.94 13.69 21.86 13.78 21.77C13.86 21.69 14.02 22.4 13.93 21.45C13.83 20.5 13.12 17.64 13.19 16.07C13.26 14.5 14.16 13.35 14.36 12.04C14.56 10.74 14.25 8.78 14.38 8.23C14.51 7.68 15.11 8.42 15.14 8.74C15.16 9.06 14.6 9.83 14.51 10.14C14.42 10.44 14.53 10.46 14.61 10.57C14.68 10.69 14.78 10.83 14.95 10.84C15.12 10.85 15.37 10.97 15.63 10.63C15.89 10.28 16.38 9.15 16.51 8.76C16.65 8.36 16.52 8.4 16.46 8.25C16.4 8.1 16.58 8.21 16.16 7.87C15.73 7.53 14.4 6.53 13.91 6.21C13.41 5.89 13.44 5.98 13.19 5.96Z",
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
