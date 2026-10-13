"""Render every scene, then stitch them into one MP4 with soft subtitles.

Usage (run build_audio.py first):
    python build_video.py --quality h --media-dir <scratch dir> --out p-value-explained.mp4

--quality is Manim's: l (480p15), m (720p30), h (1080p60), k (4K60).
Scenes render in parallel (--jobs); per-scene .srt files from Manim's
subcaptions are offset and merged into one .srt next to the output.
"""

import argparse
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

from narration import SCENE_ORDER

HERE = Path(__file__).resolve().parent
FOLDER = {"l": "480p15", "m": "720p30", "h": "1080p60", "p": "1440p60", "k": "2160p60"}
SRT_TIME = re.compile(r"(\d+):(\d+):(\d+),(\d+)")


def render(scene, quality, media_dir):
    # One media dir per scene: parallel renders sharing a Tex cache race on its .dvi files.
    cmd = [sys.executable, "-m", "manim", f"-q{quality}", "--media_dir", str(media_dir / scene), str(HERE / "pvalue_scenes.py"), scene]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return scene, f"{result.stdout[-4000:]}\n{result.stderr[-4000:]}"
    return scene, None


def duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def parse_ts(text):
    h, m, s, ms = map(int, SRT_TIME.match(text).groups())
    return timedelta(hours=h, minutes=m, seconds=s, milliseconds=ms)


def format_ts(t):
    ms = int(round(t.total_seconds() * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def merge_srt(parts):
    """parts: list of (srt_path, offset_seconds) -> merged SRT text."""
    entries = []
    for path, offset in parts:
        if not path.exists():
            continue
        for block in path.read_text(encoding="utf-8").strip().split("\n\n"):
            lines = block.strip().splitlines()
            if len(lines) < 3:
                continue
            start, end = [parse_ts(x.strip()) for x in lines[1].split("-->")]
            shift = timedelta(seconds=offset)
            entries.append((start + shift, end + shift, "\n".join(lines[2:])))
    entries.sort(key=lambda e: e[0])
    return "\n\n".join(
        f"{i}\n{format_ts(a)} --> {format_ts(b)}\n{text}" for i, (a, b, text) in enumerate(entries, 1)
    ) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality", default="h", choices=sorted(FOLDER))
    parser.add_argument("--media-dir", type=Path, default=HERE / "media")
    parser.add_argument("--out", type=Path, default=HERE / "p-value-explained.mp4")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--scenes", nargs="*", default=SCENE_ORDER)
    parser.add_argument("--skip-render", action="store_true")
    args = parser.parse_args()

    if not args.skip_render:
        failed = []
        with ThreadPoolExecutor(args.jobs) as pool:
            for scene, error in pool.map(lambda s: render(s, args.quality, args.media_dir), args.scenes):
                print(f"{'FAILED' if error else 'rendered'} {scene}", flush=True)
                if error:
                    failed.append(scene)
                    print(error, flush=True)
        if failed:
            sys.exit(f"failed scenes: {' '.join(failed)}")

    clips = [args.media_dir / s / "videos" / "pvalue_scenes" / FOLDER[args.quality] / f"{s}.mp4" for s in SCENE_ORDER]
    durations = [duration(c) for c in clips]
    offsets = [sum(durations[:i]) for i in range(len(clips))]
    t = sum(durations)
    srt_path = args.out.with_suffix(".srt")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    srt_path.write_text(merge_srt([(c.with_suffix(".srt"), o) for c, o in zip(clips, offsets)]), encoding="utf-8")

    # Manim's audio track stops at the last sound, short of the video; pad each scene's
    # audio to its video length so the voice never drifts across scene boundaries.
    pads = "".join(f"[{i}:a]apad=whole_dur={d:.3f},atrim=duration={d:.3f}[a{i}];" for i, d in enumerate(durations))
    pairs = "".join(f"[{i}:v][a{i}]" for i in range(len(clips)))
    graph = f"{pads}{pairs}concat=n={len(clips)}:v=1:a=1[v][cat];[cat]loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000[a]"
    inputs = [arg for c in clips for arg in ("-i", str(c))]

    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            *inputs,
            "-i", str(srt_path),
            "-filter_complex", graph,
            "-map", "[v]", "-map", "[a]", "-map", f"{len(clips)}:s",
            "-c:v", "libx264", "-crf", "22", "-preset", "slow", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "160k",
            "-c:s", "mov_text", "-metadata:s:s:0", "language=eng",
            "-movflags", "+faststart",
            str(args.out),
        ],
        check=True,
    )
    print(f"wrote {args.out} ({t:.1f}s) and {srt_path}")


if __name__ == "__main__":
    main()
