#!/usr/bin/env python3
"""Generate Lumen mascot illustrations via NVIDIA NIM image models.

Reads LLM_API_KEY from the x-coach .env internally; the key is never printed.
"""
import argparse
import base64
import json
import sys
import time
from pathlib import Path

import requests

ENV_PATH = Path(r"C:\Users\ttsh1\code\x-coach\.env")


def load_key() -> str:
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith("LLM_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("LLM_API_KEY not found in .env")


def probe(key: str) -> None:
    """List models reachable with this key; also try known genai endpoints with a HEAD-ish check."""
    r = requests.get(
        "https://integrate.api.nvidia.com/v1/models",
        headers={"Authorization": f"Bearer {key}"},
        timeout=30,
    )
    print(f"integrate /v1/models -> HTTP {r.status_code}")
    if r.ok:
        ids = [m.get("id", "") for m in r.json().get("data", [])]
        img_like = [i for i in ids if any(t in i.lower() for t in ("flux", "diffusion", "image", "sdxl", "bria", "consistory"))]
        print(f"total models: {len(ids)}")
        print("image-like models:")
        for i in sorted(img_like):
            print(f"  {i}")


GENAI_ENDPOINTS = {
    "flux.1-schnell": "https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-schnell",
    "flux.1-dev": "https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-dev",
    "sd3-medium": "https://ai.api.nvidia.com/v1/genai/stabilityai/stable-diffusion-3-medium",
    "sdxl": "https://ai.api.nvidia.com/v1/genai/stabilityai/stable-diffusion-xl",
}


def extract_b64(data: dict) -> str | None:
    if "image" in data and isinstance(data["image"], str):
        return data["image"]
    arts = data.get("artifacts") or []
    if arts and isinstance(arts, list):
        return arts[0].get("base64")
    return None


def generate(key: str, model: str, prompt: str, out: Path, seed: int = 0, steps: int | None = None) -> bool:
    url = GENAI_ENDPOINTS[model]
    payload: dict = {"prompt": prompt, "seed": seed}
    if model.startswith("flux"):
        schnell = "schnell" in model
        payload.update({"mode": "base", "width": 1024, "height": 1024,
                        "cfg_scale": 0 if schnell else 3.5,
                        "steps": steps or (4 if schnell else 30)})
    elif model == "sd3-medium":
        payload.update({"cfg_scale": 5, "steps": steps or 30, "aspect_ratio": "1:1",
                        "negative_prompt": ""})
    else:  # sdxl
        payload = {"text_prompts": [{"text": prompt, "weight": 1}],
                   "cfg_scale": 5, "sampler": "K_DPM_2_ANCESTRAL", "seed": seed, "steps": steps or 25}
    headers = {
        "Authorization": f"Bearer {key}",
        "Accept": "application/json",
        "NVCF-POLL-SECONDS": "5",
    }
    # retry transient gateway/queue errors
    r = None
    for attempt in range(6):
        r = requests.post(url, headers=headers, json=payload, timeout=120)
        if r.status_code not in (429, 502, 503, 504):
            break
        wait = 15 * (attempt + 1)
        print(f"  HTTP {r.status_code} (transient), retry {attempt + 1}/5 in {wait}s...")
        time.sleep(wait)
    # NVCF long-running: 202 + request id -> poll status endpoint
    deadline = time.time() + 480
    while r.status_code == 202 and time.time() < deadline:
        req_id = r.headers.get("NVCF-REQID")
        if not req_id:
            print("  202 but no NVCF-REQID header; giving up")
            return False
        print(f"  202 queued (req {req_id[:8]}...), polling...")
        time.sleep(3)
        r = requests.get(
            f"https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/{req_id}",
            headers=headers,
            timeout=60,
        )
    print(f"{model} -> HTTP {r.status_code}")
    if not r.ok:
        print(f"  body: {r.text[:400]}")
        return False
    b64 = extract_b64(r.json())
    if not b64:
        print(f"  no image in response; keys: {list(r.json().keys())}")
        return False
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(base64.b64decode(b64))
    print(f"  saved {out} ({out.stat().st_size//1024} KB)")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--model", default="flux.1-schnell", choices=list(GENAI_ENDPOINTS))
    ap.add_argument("--prompt")
    ap.add_argument("--out")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    key = load_key()
    if args.probe:
        probe(key)
        return
    if not args.prompt or not args.out:
        raise SystemExit("--prompt and --out required")
    ok = generate(key, args.model, args.prompt, Path(args.out), seed=args.seed)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
