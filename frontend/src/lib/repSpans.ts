// Frame bookkeeping shared by the coarse and dense extraction passes (RS-SP2 spec §2.5).
//
// WHY A FUNCTION AND NOT A COUNTER. poseExtract used to number frames with an incrementing
// counter, which equals round(t * 30) only when the sampling step happens to be 1/30. The coarse
// pass steps by 3/30, so a counter would number its samples 0,1,2,… while the video's frames are
// 0,3,6,… — every rep window derived from it would land in the wrong index space, silently. Both
// passes now derive the index from the TIMESTAMP, so they share one coordinate system.

/** The grid every frame_index is expressed on, matching poseExtract's fixed sampling cadence. */
export const CANONICAL_FPS = 30;

/** The frame_index of the sample at `t` seconds. */
export function frameIndexAt(t: number): number {
  return Math.round(t * CANONICAL_FPS);
}
