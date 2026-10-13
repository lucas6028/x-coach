# What is a p-value? — an explainer video

A narrated animated explainer, about 5 minutes 20 seconds long, in the visual style of 3Blue1Brown: black
background, Computer Modern math, blue/yellow highlights, and every idea shown
as a moving picture. It is built with [Manim Community](https://www.manim.community/),
the open-source fork of Grant Sanderson's animation engine.

- `p-value-explained.mp4`: the rendered video (1080p60, AAC audio, built-in English subtitles)
- `p-value-explained.srt`: the same subtitles as a separate file

## What it covers

| # | Scene | Idea |
|---|---|---|
| 1 | Hook | A "fair" coin lands heads 62 times out of 100. Is that suspicious? |
| 2 | Null hypothesis | Assume the boring explanation, $H_0: P(\text{heads}) = 0.5$ |
| 3 | Simulation | Rerun the experiment 10,000 times in a fair-coin world; the null distribution builds up |
| 4 | Tail area | p = area at least as extreme as 62, on both tails: P(≤38) + P(≥62) ≈ 0.021 |
| 5 | Definition | $p = P(\text{data at least this extreme} \mid H_0)$, and why the bar matters |
| 6 | Threshold | α = 0.05. Moving a slider shows 60 heads (p ≈ 0.057) vs. 61 heads (p ≈ 0.035) |
| 7 | Trap 1 | $P(\text{data} \mid H_0) \ne P(H_0 \mid \text{data})$, with the English speaker / U.S. president example |
| 8 | Trap 2 | A small p is not a big effect: a 51% coin gives p ≈ 0.92 at n = 100 and p ≈ 0.047 at n = 10,000 |
| 9 | Trap 3 | A large p doesn't prove $H_0$; the honest call is *undetermined* |
| 10 | Trap 4 | Many tests: 20 fair coins give 1 − 0.95²⁰ ≈ 64% odds that at least one is "significant" |
| 11 | Summary | A p-value measures surprise under $H_0$. It's a reason to look closer, not a verdict |

Every number in the video is computed exactly from Binomial(100, ½), except the
Trap 2 curve, which uses a normal approximation with a continuity correction.
Trap 3 lines up with this repo's note convention: `p ≥ 0.05` is reported as
*undetermined*, never "no difference".

## Rebuilding

Files:

- `narration.py`: the script. Each scene is split into voice segments, and each sentence has a caption form and a spoken form.
- `build_audio.py`: voices the script offline with [Kokoro](https://github.com/thewh1teagle/kokoro-onnx) TTS. It writes `audio/*.wav` and `audio/timings.json`.
- `pvalue_scenes.py`: the 11 Manim scenes. They time their animations against `timings.json`, so the picture stays in sync with the voice.
- `build_video.py`: renders the scenes in parallel, concatenates them, and merges the subtitles.

These tools live outside the repo's `.venv`. A separate venv works best (the commands below are
POSIX; on Windows, use `manim-venv\Scripts\python.exe`):

```bash
# system: ffmpeg, a LaTeX install with dvisvgm + standalone.cls, pango/cairo dev headers
python -m venv manim-venv
manim-venv/bin/python -m pip install manim kokoro-onnx soundfile

# Kokoro model files (~350 MB), from the kokoro-onnx "model-files-v1.0" release
mkdir kokoro && cd kokoro
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
cd ..

# run from this directory
manim-venv/bin/python build_audio.py --models kokoro            # --voice am_michael by default
manim-venv/bin/python build_video.py --quality h                # writes p-value-explained.mp4 + .srt
```

Use `--quality l` (480p15) for a quick preview, which renders in about a minute.
`--scenes S04Tail S06Threshold` re-renders only the scenes you name. To change the
voice, pass any Kokoro voice id to `build_audio.py --voice`.
