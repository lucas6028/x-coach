"""Generate the voice-over for tcn_video.py with ElevenLabs text-to-speech.

Every self.say(...) call in tcn_video.py is one narration line. The voice reads
the call's `spoken=` text when it has one (captions keep their math notation,
the voice gets words), otherwise the caption itself.

Clips go to narration/clips/ and are listed in narration/manifest.json, keyed
by their text, so a re-run only pays for lines that are new or changed.
Commit the narration/ folder: regenerating it costs credits.

    python tts.py                        # dry run: pending lines and their cost
    python tts.py --generate --limit 1   # one line, to check the voice works
    python tts.py --generate             # everything still missing
    python tts.py --prune                # delete clips the script no longer uses

Reads the API key from ELEVENLABS_API_KEY. In a Claude Code cloud environment,
a network secret for api.elevenlabs.io (header xi-api-key) works instead.
Standard library only.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCENE_FILE = HERE / "tcn_video.py"
OUT_DIR = Path(os.environ.get("TCN_NARRATION_DIR", HERE / "narration"))
MANIFEST = OUT_DIR / "manifest.json"

API = os.environ.get("ELEVENLABS_API_BASE", "https://api.elevenlabs.io/v1")
DEFAULT_VOICE = "onwK4e9ZLuTAKqWW03F9"  # "Daniel - Steady Broadcaster", a premade voice
DEFAULT_MODEL = "eleven_multilingual_v2"
OUTPUT_FORMAT = "mp3_44100_128"
SEED = 1  # same text, same take (best effort on ElevenLabs' side)

# Credits per character, used only when the API's model list doesn't say.
# Unknown models are assumed to cost the most.
FALLBACK_COST = {
    "eleven_multilingual_v2": 1.0,
    "eleven_turbo_v2_5": 0.5,
    "eleven_flash_v2_5": 0.5,
}


def narration_lines(scene_file: Path = SCENE_FILE) -> list[str]:
    """The spoken text of every self.say(...) call, in source order, deduplicated."""
    tree = ast.parse(scene_file.read_text(encoding="utf-8"))
    calls = sorted(
        (
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "say"
        ),
        key=lambda n: (n.lineno, n.col_offset),
    )
    lines = []
    for call in calls:
        node = next((k.value for k in call.keywords if k.arg == "spoken"), call.args[0])
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            sys.exit(f"{scene_file.name}:{call.lineno}: say() text must be a string literal")
        lines.append(node.value)
    return list(dict.fromkeys(lines))


def clip_name(text: str, voice: str, model: str) -> str:
    digest = hashlib.sha1(f"{voice}|{model}|{OUTPUT_FORMAT}|{text}".encode()).hexdigest()
    return f"clips/{digest[:16]}.mp3"


def load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {"clips": {}}


def save_manifest(manifest: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(MANIFEST)


def pending(lines: list[str], manifest: dict, voice: str, model: str) -> list[str]:
    """Lines without a clip for this voice and model."""
    have = manifest.get("clips", {})
    return [
        t for t in lines
        if have.get(t) != clip_name(t, voice, model) or not (OUT_DIR / have[t]).exists()
    ]


def api(path: str, key: str | None, body: dict | None = None) -> bytes:
    # With no key, send none: a proxy holding the key (e.g. a Claude Code
    # network secret) adds it on the way out.
    headers = {"Content-Type": "application/json"}
    if key:
        headers["xi-api-key"] = key
    req = urllib.request.Request(
        f"{API}{path}",
        data=None if body is None else json.dumps(body).encode(),
        method="GET" if body is None else "POST",
        headers=headers,
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")
            busy = e.code == 429 and "quota" not in detail
            if (busy or e.code >= 500) and attempt < 3:
                time.sleep(2 ** (attempt + 1))
                continue
            raise RuntimeError(f"HTTP {e.code} from {path}: {detail}") from None
        except urllib.error.URLError as e:
            raise RuntimeError(f"can't reach {API}: {e.reason}") from None
    raise AssertionError("unreachable")


def credits_left(key: str | None) -> int | None:
    """Remaining character credits, or None if the key may not read the subscription."""
    try:
        sub = json.loads(api("/user/subscription", key))
    except RuntimeError as e:
        print(f"note: couldn't read your subscription ({e})")
        return None
    return sub["character_limit"] - sub["character_count"]


def cost_per_char(key: str | None, model: str) -> float:
    try:
        for m in json.loads(api("/models", key)):
            rate = (m.get("model_rates") or {}).get("character_cost_multiplier")
            if m.get("model_id") == model and rate is not None:
                return float(rate)
    except RuntimeError as e:
        print(f"note: couldn't read the model list ({e})")
    return FALLBACK_COST.get(model, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generate", action="store_true", help="call the API (otherwise: dry run)")
    ap.add_argument("--limit", type=int, help="generate at most this many lines")
    ap.add_argument("--voice", default=DEFAULT_VOICE)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-credits", type=int, help="refuse to spend more than this")
    ap.add_argument("--prune", action="store_true", help="delete clips no longer used by the script")
    args = ap.parse_args()

    lines = narration_lines()
    manifest = load_manifest()
    todo = pending(lines, manifest, args.voice, args.model)
    if args.limit is not None:
        todo = todo[: args.limit]

    if args.prune:
        keep = {clip_name(t, args.voice, args.model) for t in lines}
        manifest["clips"] = {t: f for t, f in manifest["clips"].items() if t in lines and f in keep}
        for mp3 in (OUT_DIR / "clips").glob("*.mp3"):
            if f"clips/{mp3.name}" not in keep:
                mp3.unlink()
                print(f"deleted {mp3.name}")
        save_manifest(manifest)

    key = os.environ.get("ELEVENLABS_API_KEY")
    rate = cost_per_char(key, args.model)
    chars = sum(len(t) for t in todo)
    cost = math.ceil(chars * rate)
    print(
        f"{len(lines)} narration lines; {len(todo)} to generate with {args.model}: "
        f"{chars} characters, about {cost} credits at {rate:g}/character"
    )
    left = credits_left(key)
    if left is not None:
        print(f"{left} credits left on your plan")
    if not args.generate or not todo:
        return
    if left is None and not key:
        sys.exit("no API key: set ELEVENLABS_API_KEY, or add a network secret for api.elevenlabs.io")

    if left is not None:
        if cost > left:
            sys.exit("not enough credits; use --limit, or a cheaper --model")
    elif args.max_credits is None:
        sys.exit("can't check your remaining credits; pass --max-credits to set a ceiling")
    if args.max_credits is not None and cost > args.max_credits:
        sys.exit(f"would spend about {cost} credits, over --max-credits {args.max_credits}")

    (OUT_DIR / "clips").mkdir(parents=True, exist_ok=True)
    manifest.update(voice_id=args.voice, model_id=args.model, output_format=OUTPUT_FORMAT)
    manifest.setdefault("clips", {})
    for i, text in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {text[:70]}{'…' if len(text) > 70 else ''}")
        try:
            audio = api(
                f"/text-to-speech/{args.voice}?output_format={OUTPUT_FORMAT}",
                key,
                {"text": text, "model_id": args.model, "seed": SEED},
            )
        except RuntimeError as e:
            save_manifest(manifest)
            sys.exit(f"stopped: {e}")
        rel = clip_name(text, args.voice, args.model)
        (OUT_DIR / rel).write_bytes(audio)
        manifest["clips"][text] = rel
        save_manifest(manifest)  # after every clip, so an interruption loses nothing paid for

    after = credits_left(key)
    if left is not None and after is not None:
        print(f"done: used {left - after} credits, {after} left")


if __name__ == "__main__":
    main()
