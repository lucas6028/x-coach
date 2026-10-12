#!/usr/bin/env bash
# Render every scene in tcn_video.py and join them into one MP4.
#
#   ./render.sh            final cut, 1920x1080 @ 30 fps  -> tcn-explained.mp4
#   ./render.sh preview    fast draft, 854x480 @ 15 fps   -> tcn-explained-preview.mp4
#
# Env: MANIM (manim executable, default "manim"), JOBS (parallel renders, default 4).
set -euo pipefail
cd "$(dirname "$0")"

MANIM=${MANIM:-manim}
JOBS=${JOBS:-4}
MEDIA="$PWD/media"

if [[ "${1:-}" == "preview" ]]; then
  FLAGS=(-ql)
  SUBDIR=480p15
  OUT=tcn-explained-preview.mp4
else
  FLAGS=(-r 1920,1080 --fps 30)
  SUBDIR=1080p30
  OUT=tcn-explained.mp4
fi

# Scene classes are named S01_..., S02_..., so file order is play order.
mapfile -t SCENES < <(grep -oE '^class S[0-9]+_[A-Za-z0-9]+' tcn_video.py | cut -d' ' -f2)
mkdir -p "$MEDIA/logs"

running=0
for scene in "${SCENES[@]}"; do
  echo "rendering $scene"
  # One media dir per scene: concurrent renders otherwise race on Manim's shared LaTeX cache.
  "$MANIM" render "${FLAGS[@]}" --media_dir "$MEDIA/$scene" tcn_video.py "$scene" >"$MEDIA/logs/$scene.log" 2>&1 &
  running=$((running + 1))
  if ((running >= JOBS)); then
    wait -n || true
    running=$((running - 1))
  fi
done
wait || true

LIST="$MEDIA/concat.txt"
: >"$LIST"
for scene in "${SCENES[@]}"; do
  clip="$MEDIA/$scene/videos/tcn_video/$SUBDIR/$scene.mp4"
  if [[ ! -f "$clip" ]]; then
    echo "missing $clip; see $MEDIA/logs/$scene.log" >&2
    exit 1
  fi
  echo "file '$clip'" >>"$LIST"
done

ffmpeg -y -loglevel error -f concat -safe 0 -i "$LIST" -c copy -movflags +faststart "$OUT"
echo "wrote $OUT"
