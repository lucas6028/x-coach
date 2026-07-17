// The browsable movement catalog. Values MUST match the backend's canonical spelling verbatim
// (they are passed straight through as the `movement` scope on /api/knowledge/graph).
export const FLAGSHIP_MOVEMENTS = ["Squat", "Lunge", "Push-up", "Overhead Press", "Row"] as const;
export const GENERAL_MOVEMENTS = [
  "Deadlift",
  "Bicep Curl",
  "Band Pull Apart",
  "Arm Abduction",
  "Arm VW",
  "Leg Abduction",
  "Sit-up",
  "Shoulder Bridge",
  "Jumping Jacks",
  "High Knee",
  "Torso Twist",
] as const;
export const ALL_MOVEMENTS = [...FLAGSHIP_MOVEMENTS, ...GENERAL_MOVEMENTS];
export type Movement = (typeof ALL_MOVEMENTS)[number];
