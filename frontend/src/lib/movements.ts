// The browsable movement catalog. Values MUST match the backend's canonical spelling verbatim
// (they are the `movement` node attribute in sports_kg_v3 and the `movement` scope on
// /api/knowledge/*).
//
// Grouped by body region rather than by the graph's flagship/general depth tiers: the tiers say how
// much knowledge we authored, which is our concern, not the reader's. Someone picking what to train
// reasons in body regions.
const LOWER_BODY = ["Squat", "Lunge", "Deadlift", "Leg Abduction", "Shoulder Bridge"] as const;
const UPPER_BODY = [
  "Push-up",
  "Overhead Press",
  "Row",
  "Bicep Curl",
  "Band Pull Apart",
  "Arm Abduction",
  "Arm VW",
] as const;
const CORE = ["Sit-up", "Torso Twist"] as const;
const FULL_BODY = ["Jumping Jacks", "High Knee"] as const;

export const MOVEMENT_GROUPS = [
  { key: "movements.groupLower", items: LOWER_BODY },
  { key: "movements.groupUpper", items: UPPER_BODY },
  { key: "movements.groupCore", items: CORE },
  { key: "movements.groupFullBody", items: FULL_BODY },
] as const;

export const ALL_MOVEMENTS = [...LOWER_BODY, ...UPPER_BODY, ...CORE, ...FULL_BODY] as const;
export type Movement = (typeof ALL_MOVEMENTS)[number];

// Which movements the pipeline can analyse is NOT stated here. It comes from GET /api/movements,
// which derives it from the Python detector registry, so registering a fourth detector surfaces
// it in the UI with no frontend edit. The previous hand-maintained ANALYZABLE_MOVEMENTS constant
// was a second list that had to be kept in sync by hand.
export interface AnalyzableMovement {
  name: string;
  /** False when the rules are literature-derived but never checked against labeled ground
   *  truth — rendered with a Beta tag. */
  validated: boolean;
}
