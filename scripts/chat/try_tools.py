"""Drive the coach's tool-calling loop end-to-end from the command line.

Calls ``chat_service.answer_stream`` directly, so it exercises the REAL loop, the REAL tools
(live KG + RAG on disk) and the REAL model — but skips the router, so no Supabase session, no
browser, and no analysis upload are needed. Every SSE frame is printed as it arrives, and each
tool call is shown with the arguments the model actually chose, which is the thing you cannot
see from the tray.

Run from the repository root:

    .venv\\Scripts\\python.exe scripts/chat/try_tools.py "我第 2 rep 的膝蓋角度是多少?"
    .venv\\Scripts\\python.exe scripts/chat/try_tools.py --clean "文獻上怎麼說腳踝活動度?"
    .venv\\Scripts\\python.exe scripts/chat/try_tools.py --unmeasured "我的深蹲怎麼樣?"
    .venv\\Scripts\\python.exe scripts/chat/try_tools.py --model openai/gpt-oss-20b "..."

Contexts (`--clean` / `--unmeasured` select the last two):

* default     — one detected fault, with the full detail blob the client ships
* --clean     — zero faults on a measurable clip (the CLEAN REP branch)
* --unmeasured— zero faults because NOTHING was measurable (the NOT MEASURED branch)

The last one is the honesty test that matters most: the coach must not turn a `rag_search`
lookup into feedback about a rep that never measured.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# The Windows console defaults to cp950 here, which cannot encode characters the models routinely
# emit (narrow no-break space, curly quotes, CJK punctuation) — without this the script dies mid
# answer with UnicodeEncodeError and you lose the very output you were testing.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):  # pragma: no cover — non-reconfigurable stream
        pass

from backend.app.services import chat as chat_service  # noqa: E402
from backend.app.settings import resolve_chat_model  # noqa: E402

# One detected fault, shaped exactly as ``buildChatContext`` builds it (frontend/src/lib/grounding.ts):
# the compact ``faults`` summary the prompt is built from, plus the uncompressed ``detail`` blob that
# only ``get_analysis`` can reach. The split is the whole point — anything in ``detail`` but not in
# ``faults`` is a question the coach can ONLY answer by calling the tool.
_DETECTION: dict[str, Any] = {
    "fault_id": "squat_insufficient_depth",
    "fault_name": "Insufficient Depth",
    "kg_query": "insufficient squat depth",
    "retrieval_mode": "kg",
    "severity": 0.72,
    "confidence": 0.88,
    "observability": "clear",
    "start_time": 3.10,
    "end_time": 4.05,
    "start_frame": 93,
    "end_frame": 121,
    "peak_frame": 108,
    "phase": "bottom",
    # The FULL measured dict. The prompt only ever sees the one-line summary in ``faults`` below.
    "evidence": {
        "hip_knee_delta_deg": 12.4,
        "min_knee_angle_deg": 104.7,
        "hip_depth_ratio": 0.83,
        "torso_lean_deg": 41.2,
    },
    # Per-rep attribution — absent from the compact blob, so "which rep?" forces a get_analysis call.
    "rep_index": 2,
    "occurred_reps": [2],
    "rep_count": 1,
}

_RETRIEVAL: dict[str, Any] = {
    "fault_id": "squat_insufficient_depth",
    "fault_name": "Insufficient Depth",
    "query_text": "insufficient squat depth",
    "retrieval_mode": "kg",
    "context": {
        "matched_nodes": ["Insufficient Depth"],
        "results": [
            {
                "rank": 1,
                "score": 0.71,
                "text": (
                    "Squat depth is commonly assessed by the hip crease relative to the knee. "
                    "Restricted ankle dorsiflexion and limited hip flexion are frequent limiters."
                ),
                "metadata": {"source": "example-review.pdf", "page": 4},
            }
        ],
    },
}

_FAULT_SUMMARY: dict[str, Any] = {
    "fault_name": "Insufficient Depth",
    "phase": "bottom",
    "severity": 0.72,
    "start_time": 3.10,
    "end_time": 4.05,
    "evidence": "hip_knee_delta_deg 12.4",  # the SHORT string keyEvidence() produces
    "causes": ["limited ankle dorsiflexion", "limited hip flexion"],
    "risks": ["reduced training stimulus"],
    "corrections": ["elevate the heels", "practise a paused bodyweight squat"],
    "rag_snippet": "Squat depth is commonly assessed by the hip crease relative to the knee.",
}

_DETAIL: dict[str, Any] = {
    "metadata": {"fps": 30.0, "width": 1080, "height": 1920, "total_frames": 300},
    "quality": {"valid_frame_ratio": 0.94, "lower_body_visibility_mean": 0.91},
    "view": {"view_type": "side", "view_confidence": 0.93},
    "detections": [_DETECTION],
    "retrievals": [_RETRIEVAL],
}


def _context(kind: str) -> dict[str, Any]:
    base: dict[str, Any] = {
        "video_id": "cli-probe",
        "movement": "Squat",
        "view_type": "side",
        "view_confidence": 0.93,
    }
    if kind == "clean":
        # Measurable, but nothing detected — the CLEAN REP branch.
        return {
            **base,
            "fault_count": 0,
            "quality": {"valid_frame_ratio": 0.94, "lower_body_visibility_mean": 0.91},
            "faults": [],
            "detail": {**_DETAIL, "detections": [], "retrievals": []},
        }
    if kind == "unmeasured":
        # Zero measurable frames — the NOT MEASURED branch. An empty fault list here does NOT
        # mean good form, and the coach must not let a tool lookup become feedback anyway.
        return {
            **base,
            "fault_count": 0,
            "quality": {"valid_frame_ratio": 0.0, "lower_body_visibility_mean": 0.12},
            "faults": [],
            "detail": {
                **_DETAIL,
                "quality": {"valid_frame_ratio": 0.0, "lower_body_visibility_mean": 0.12},
                "detections": [],
                "retrievals": [],
            },
        }
    return {
        **base,
        "fault_count": 1,
        "quality": {"valid_frame_ratio": 0.94, "lower_body_visibility_mean": 0.91},
        "faults": [_FAULT_SUMMARY],
        "detail": _DETAIL,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question", help="the user turn to send")
    ap.add_argument("--model", default=None, help="model slug (default: the server default)")
    ap.add_argument("--clean", action="store_true", help="use a clean-rep context (0 faults, measurable)")
    ap.add_argument("--unmeasured", action="store_true", help="use an unmeasurable-clip context")
    ap.add_argument("--show-prompt", action="store_true", help="print the system prompt and exit")
    args = ap.parse_args()

    kind = "clean" if args.clean else ("unmeasured" if args.unmeasured else "fault")
    context = _context(kind)
    model = resolve_chat_model(args.model)

    if args.show_prompt:
        print(chat_service._build_system_prompt(context) + chat_service._TOOL_GROUNDING_RULE)
        return 0

    print(f"context : {kind}")
    print(f"model   : {model}")
    print(f"question: {args.question}\n")

    tools_called: list[str] = []
    answer: list[str] = []
    started = time.monotonic()
    first_token_at: float | None = None

    for frame in chat_service.answer_stream(
        messages=[{"role": "user", "content": args.question}], context=context, model=model
    ):
        event = frame.split("\n", 1)[0][len("event:") :].strip()
        data = json.loads(frame.split("data:", 1)[1].strip())
        if event == "delta":
            if first_token_at is None:
                first_token_at = time.monotonic() - started
            answer.append(data["text"])
            print(data["text"], end="", flush=True)
        elif event == "tool":
            tools_called.append(data["name"])
            print(f"\n  \033[36m[tool] {data['name']}({data['query']!r})\033[0m", flush=True)
        elif event == "tool_done":
            n = len(data.get("sources", []))
            print(f"  \033[36m[done] {n} sources\033[0m", flush=True)
        elif event == "reset":
            # The round narrated AND called a tool; everything printed above was retracted.
            answer.clear()
            first_token_at = None
            print("\n  \033[33m[reset] ^^ narration above was RETRACTED by the server\033[0m", flush=True)
        elif event == "error":
            print(f"\n\n  \033[31m[error] {data['detail']}\033[0m")
        elif event == "done":
            elapsed = time.monotonic() - started
            ttft = f"{first_token_at:.1f}s" if first_token_at is not None else "n/a"
            print(f"\n\n  \033[32m[done]\033[0m model={data['model']} total={elapsed:.1f}s ttft={ttft}")

    print(f"\ntools called: {tools_called or 'NONE — the model answered from the prompt alone'}")
    if not tools_called:
        print("  (if you expected a tool, the question was answerable from the compact blob —")
        print("   ask for something only the full analysis or the literature could supply)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
