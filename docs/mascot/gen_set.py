#!/usr/bin/env python3
"""Batch-generate the Lumen expression + pose sets in the two chosen styles.

C soft-3D (seed 33) = primary bible set; A kawaii sticker (seed 7) = sticker set.
"""
from pathlib import Path

from gen_lumen import generate, load_key

SP = Path(__file__).parent

# CORE without the expression/pose phrase (inserted per variant).
BASE = (
    "cute mascot design, small round flame-spirit blob creature, not human not animal "
    "no ears, candle-flame golden fire tuft on top of round head, golden-yellow sports "
    "sweatband on forehead, small gold coach whistle on red cord around neck, tiny navy "
    "sneakers, big round navy eyes with sparkle highlights, golden blush, stubby arms, "
    "warm gold cream navy palette, soft golden glow, {mood}, full body, "
    "plain warm-cream background, single character, "
)

STYLE_C = ("modern flat vector app mascot, minimal geometric shapes, no outlines, "
           "solid flat colors with single-tone shadow shapes, clean contemporary "
           "tech-brand mascot design, crisp edges.")
STYLE_A = ("soft kawaii chibi sticker art, thick smooth clean outlines, flat cel "
           "shading, rounded simple shapes, adorable and huggable, high quality 2D "
           "character art.")

MOODS = {
    "genki":   "cheerful open smile, front view, one arm waving hello",
    "think":   "curious thinking face, head tilted, one arm touching chin",
    "proud":   "warm content smile, happy curved eyes, hands together",
    "scan":    "focused analyzing look, holding a golden magnifying glass up to one eye",
    "coach":   "encouraging smile, pointing forward with one arm, coaching stance",
    "whistle": "puffed cheeks blowing the gold whistle, one arm raised high",
    "cheer":   "jumping in celebration, both arms up, joyful open-mouth laugh, sparkles",
}

JOBS = [("C", STYLE_C, 33, ["genki", "think", "proud", "scan", "coach", "whistle", "cheer"]),
        ("A", STYLE_A, 7,  ["genki", "think", "proud", "cheer"])]


def main() -> None:
    key = load_key()
    ok = fail = 0
    for tag, style, seed, moods in JOBS:
        for mood in moods:
            out = SP / f"set-{tag}-{mood}.png"
            if out.exists():
                print(f"skip {out.name} (exists)", flush=True)
                ok += 1
                continue
            prompt = BASE.format(mood=MOODS[mood]) + style
            print(f"--- {tag}/{mood} (seed {seed}, {len(prompt)} chars)", flush=True)
            if generate(key, "flux.1-dev", prompt, out, seed=seed):
                ok += 1
            else:
                fail += 1
    print(f"DONE ok={ok} fail={fail}", flush=True)


if __name__ == "__main__":
    main()
