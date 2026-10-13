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

has_audio() {
  [[ -n "$(ffprobe -v error -select_streams a -show_entries stream=index -of csv=p=0 "$1")" ]]
}

clips=()
narrated=false
for scene in "${SCENES[@]}"; do
  clip="$MEDIA/$scene/videos/tcn_video/$SUBDIR/$scene.mp4"
  if [[ ! -f "$clip" ]]; then
    echo "missing $clip; see $MEDIA/logs/$scene.log" >&2
    exit 1
  fi
  clips+=("$clip")
  if has_audio "$clip"; then narrated=true; fi
done

LIST="$MEDIA/concat.txt"
: >"$LIST"
for clip in "${clips[@]}"; do
  # With narration, every clip needs an audio track or the joined audio drifts;
  # a scene with no generated lines gets silence (Manim writes 48 kHz stereo AAC).
  if $narrated && ! has_audio "$clip"; then
    padded="${clip%.mp4}.silent-audio.mp4"
    ffmpeg -y -loglevel error -i "$clip" -f lavfi -i anullsrc=r=48000:cl=stereo \
      -shortest -c:v copy -c:a aac "$padded"
    clip=$padded
  fi
  echo "file '$clip'" >>"$LIST"
done

ffmpeg -y -loglevel error -f concat -safe 0 -i "$LIST" -c copy -movflags +faststart "$OUT"
echo "wrote $OUT"
