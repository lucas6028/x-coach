// ONE definition of "was this clip actually measured?", shared by every surface that would
// otherwise read a clean verdict out of an empty detection list.
//
// THE AMBIGUITY THIS EXISTS TO RESOLVE. `run_detector` returns an empty `detections` list
// identically for "no faults found" and "no frame was ever measurable" — a clip whose pose
// extraction produced no valid frames is byte-for-byte indistinguishable, at the detection layer,
// from a flawless rep. Three surfaces used to render a clean verdict off `detections.length`
// alone: the metrics HUD (KeyMetricsCard, and StudioMobile on the phone), the coaching tray's
// clean-rep banner (CoachTray), and the chat system prompt's "This is a CLEAN REP … congratulate the user" instruction
// (backend/app/services/chat.py). On a knees-up-cropped clip all three congratulated the user on
// form nothing had measured.
//
// WHY A SHARED HELPER RATHER THAN THE SAME EXPRESSION IN TWO COMPONENTS. `KeyMetricsCard` and
// `CoachTray` co-render on the same screen (App.tsx), so two independently-drifting definitions
// would show the user a HUD and a banner that disagree — which is a worse failure than the one
// being fixed. The backend prompt necessarily carries its own implementation (different runtime),
// and its docstring cross-references this file so the criterion stays one criterion.
//
// THE CRITERION IS CATEGORICAL — exactly zero valid frames — and deliberately NOT a
// low-but-nonzero band. No threshold for "enough frames to trust a verdict" has been measured
// anywhere in this repo, and inventing one for a user-facing verdict is precisely the kind of
// unbacked number this project exists not to emit. KNOWN, STATED GAP: a clip with one valid frame
// still reads as measured.
//
// A MISSING field counts as UNMEASURED. The analyze pipeline always emits `valid_frame_ratio`
// (src/pose/pose_rule_detector.py), so absence means the payload did not say — and "we cannot
// tell" must not resolve to "everything is fine".
export function wasMeasured(quality: Record<string, number> | null | undefined): boolean {
  return (quality?.valid_frame_ratio ?? 0) > 0;
}

/**
 * The "valid frames" figure, with the denominator RS-SP2 made correct.
 *
 * WHY IT IS NOT `valid_frame_ratio`. That ratio counts valid frames against the WHOLE CLIP. Since
 * RS-SP2 the browser deliberately extracts only the selected reps (`lib/poseExtract`), so on a
 * five-rep clip with three reps scored roughly 60% of the frames carry no landmarks BY DESIGN.
 * Divided by the clip, that reads as "30% — bad tracking": a pipeline decision presented to the
 * user as a measurement problem, on the one card whose job is to say how much was measured.
 *
 * Analyses predating SP2, and anything the CLI produced, carry no `extracted_frames` and keep the
 * original whole-clip denominator — absent is absent, not zero.
 *
 * SHARED for the same reason `wasMeasured` is: the desktop Key-metrics card and the phone metrics
 * list show the same figure side by side across viewports, and two copies of this expression would
 * drift into disagreeing about the same clip. (It carries the logic MetricsCards held before the
 * muse-spark redesign deleted that component.)
 */
export function validFrameStat(quality: Record<string, number> | null | undefined): {
  ratio: number;
  valid: number;
  extracted: number;
} {
  const valid = quality?.valid_frames ?? 0;
  const extracted = quality?.extracted_frames ?? 0;
  return {
    ratio: extracted > 0 ? valid / extracted : quality?.valid_frame_ratio ?? 0,
    valid,
    extracted,
  };
}
