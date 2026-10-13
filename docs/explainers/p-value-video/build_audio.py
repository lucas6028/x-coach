"""Voice the narration with Kokoro (offline TTS) and record segment timings.

Usage:
    python build_audio.py --models <dir with kokoro-v1.0.onnx + voices-v1.0.bin>

Writes audio/<Scene>_<i>.wav (one file per segment) and audio/timings.json,
which the Manim scenes read to time animations against the voice and to emit
subtitles. Model files: github.com/thewh1teagle/kokoro-onnx releases
(tag model-files-v1.0).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from kokoro_onnx import Kokoro

from narration import NARRATION

HERE = Path(__file__).resolve().parent
AUDIO_DIR = HERE / "audio"

LEAD_IN = 0.05  # seconds of silence before the first sentence of a segment
SENTENCE_GAP = 0.32  # pause between sentences inside a segment


def split_sentence(sentence):
    if isinstance(sentence, tuple):
        return sentence
    return sentence, sentence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", required=True, type=Path)
    parser.add_argument("--voice", default="am_michael")
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()

    kokoro = Kokoro(str(args.models / "kokoro-v1.0.onnx"), str(args.models / "voices-v1.0.bin"))
    AUDIO_DIR.mkdir(exist_ok=True)
    timings = {}

    for scene, segments in NARRATION.items():
        timings[scene] = []
        for i, sentences in enumerate(segments):
            sample_rate = 24000
            chunks = [np.zeros(int(LEAD_IN * sample_rate), dtype=np.float32)]
            cursor = LEAD_IN
            subs = []
            for j, sentence in enumerate(sentences):
                caption, spoken = split_sentence(sentence)
                samples, sample_rate = kokoro.create(spoken, voice=args.voice, speed=args.speed, lang="en-us")
                samples = np.asarray(samples, dtype=np.float32)
                duration = len(samples) / sample_rate
                subs.append({"text": caption, "offset": round(cursor, 3), "duration": round(duration, 3)})
                chunks.append(samples)
                cursor += duration
                if j < len(sentences) - 1:
                    chunks.append(np.zeros(int(SENTENCE_GAP * sample_rate), dtype=np.float32))
                    cursor += SENTENCE_GAP
            name = f"{scene}_{i}.wav"
            audio = np.concatenate(chunks)
            sf.write(AUDIO_DIR / name, audio, sample_rate)
            timings[scene].append({"file": name, "duration": round(len(audio) / sample_rate, 3), "subs": subs})
            print(f"{name}: {len(audio) / sample_rate:.2f}s")

    (AUDIO_DIR / "timings.json").write_text(json.dumps(timings, indent=2, ensure_ascii=False))
    total = sum(seg["duration"] for segs in timings.values() for seg in segs)
    print(f"total narration: {total:.1f}s")


if __name__ == "__main__":
    main()
