#!/usr/bin/env python
"""Structural linter for x-coach experiment notes (notes/*.md).

WHAT THIS CHECKS: only mechanically-decidable structure — that the note has a
claim-shaped title, a lead conclusion, an honesty/limits section, a reproduction
pointer, and a link to its pre-registration plan when one exists on disk.

WHAT THIS CANNOT CHECK: register (是否寫給人看), whether jargon was actually
expanded, whether a claim traces to a measured number, or whether a rewrite
preserved every figure. Those are the substance of the skill and live in
SKILL.md. A PASS here does NOT mean the note is good.

Usage (from repo root):
    .venv/Scripts/python.exe .claude/skills/write-experiment-note/check_note.py notes/foo.md
    .venv/Scripts/python.exe .claude/skills/write-experiment-note/check_note.py notes/*.md
    .venv/Scripts/python.exe .claude/skills/write-experiment-note/check_note.py --numbers notes/foo.md
    <PostToolUse payload on stdin> | ... check_note.py --hook

`--numbers` set-diffs the numeric tokens of the git HEAD version against the
working tree, ignoring section numbers (`§7.2`, `## 3.1`) so a restructuring
rewrite does not drown the signal. Use it after rewriting an existing note: a
rewrite for readability must not drop a single measured figure.

`--hook` is the PostToolUse entry point wired up in .claude/settings.json. It
reads the tool payload on stdin, lints only notes/*.md, always exits 0, and
reports findings as additionalContext. It never blocks a write.

Exit code: 0 if every file passes the structural checks (warnings do not fail).
"""

from __future__ import annotations

import collections
import json
import os
import re
import subprocess
import sys

# --- conventions, derived from the notes/ corpus (not from taste) -------------
# Each pattern family below was checked against the existing notes; the variants
# are the ones the corpus actually uses, in both Chinese and English notes.

# A title that states the question or the finding, not "Experiment 3 results".
TITLE_CLAIM_MARK = re.compile(r"[?？—–:：]")
TITLE_LABEL_ONLY = re.compile(r"^(結果|摘要|實驗摘要|results|summary|notes?)$", re.I)

# A lead block that fronts the answer/question before the method. The corpus does
# this two ways: an emphasised paragraph between the H1 and the first `##`
# (**Question.** / **Goal.** / **Verdict:** / **這份文件在回答什麼**), or a first
# section that is itself the lead (`## 一句話`, `## 0. 事前登錄`, `## 兩句話`).
LEAD_HEADING_PAT = re.compile(
    r"^##\s*(0[\.、\s]|一句話|兩句話|三句話|TL;?DR|結論|Conclusion|Summary|Verdict)",
    re.M | re.I,
)
LEAD_BLOCK_PAT = re.compile(r"(\*\*[^*\n]+\*\*|^>\s)", re.M)

# A section that states what the result does NOT support.
LIMITS_PAT = re.compile(
    r"(不能說|未測|沒做的事|尚未做的事|未涵蓋|限制|caveat|confound|cannot|not supported"
    r"|does not support|未解|open risk|不可宣稱)",
    re.I,
)

# A pointer that lets someone re-run it.
REPRO_PAT = re.compile(r"(重現|重跑|Reproduce|產物|artifacts?|scripts/|data/)", re.I)

# --- warning-level patterns ---------------------------------------------------

WIKILINK_PAT = re.compile(r"\[\[[^\]]+\]\]")

# A bare cross-file section reference: "§8", "計畫 §7.2". Fine if the note also
# spells out what the rule says; this is a nudge, not a rule.
BARE_SECTION_PAT = re.compile(r"§\s*\d+(\.\d+)*")

# The undetermined-vs-equivalence trap: a non-significant p described as "no
# difference" or "equivalent". Warning only — this one over-fires by design.
EQUIV_TRAP_PAT = re.compile(
    r"(沒有差異|沒有分別|兩者相同|等價|equivalen|no difference|identical performance)"
)
PVALUE_PAT = re.compile(r"p\s*[=≥>]\s*0?\.\d+", re.I)

NUMBER_PAT = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?(?![\w])", re.I)

# Tokens that are structure, not measurements: "§7.2", "## 7.2 讀法", "附錄 C.1".
# A restructuring rewrite renumbers these freely, so they must not count as
# dropped figures.
STRUCTURE_PAT = re.compile(r"(§\s*\d+(?:\.\d+)*|^#{1,6}\s*\d+(?:\.\d+)*[.、]?)", re.M)


def read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def first_h1(text: str) -> str | None:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return None


def plan_sibling(path: str) -> str | None:
    """If notes/<stem>_validation_plan.md (or <stem>_plan.md) exists, return it."""
    directory, base = os.path.split(path)
    stem = base[:-3] if base.endswith(".md") else base
    for suffix in ("_validation_plan.md", "_plan.md"):
        for candidate_stem in (stem, stem.replace("_results", "").replace("-results", "")):
            candidate = os.path.join(directory, candidate_stem + suffix)
            if os.path.exists(candidate):
                return os.path.basename(candidate)
    return None


# Home-made Chinese translations of terms of art. Advisory: the English term is what a
# reader greps, cites, and searches the literature for; a coinage breaks all three. Only
# words that are NOT standard Chinese prose are listed -- 受試者, 錄影, 做對／做錯 are fine,
# and single characters (折, 臂) or words with ordinary meanings (混淆, 事前登錄) are left
# out because they fire on prose (手臂, 混淆矩陣) far more often than on coinages.
TRANSLATED_JARGON = {
    "類內位置": "within-class position",
    "位置平衡": "position-balanced",
    "殘差化": "residualization",
    "正控制": "positive control",
    "負控制": "negative control",
    "探針": "probe",
    "漂移": "drift",
    "虛無分布": "null distribution",
    "置換檢定": "permutation test",
    "捷徑": "shortcut",
    "地板線": "floor",
    "取景": "framing",
    "判讀表": "decision table",
}


def check_file(path: str) -> tuple[list[str], list[str]]:
    text = read(path)
    head = "\n".join(text.splitlines()[:45])
    fails: list[str] = []
    warns: list[str] = []

    title = first_h1(text)
    if not title:
        fails.append("no H1 title")
    else:
        bare = re.split(r"[—–:：]", title)[0].strip()
        if not TITLE_CLAIM_MARK.search(title) or TITLE_LABEL_ONLY.match(bare):
            # Warning, not a failure: walkthrough/reference notes in this corpus
            # legitimately use descriptive titles. For a RESULTS note, the corpus
            # convention is to state the finding or the question.
            warns.append(
                f"H1 reads as a label rather than a claim or question: {title!r} — "
                "for a results note the corpus states the finding in the title, e.g. "
                "'B1 repeated splits:person-crop 的 +0.026 是分割雜訊'"
            )

    lead_zone = text.split("\n## ", 1)[0]
    if not (LEAD_HEADING_PAT.search(head) or LEAD_BLOCK_PAT.search(lead_zone)):
        fails.append(
            "nothing fronts the answer: expected either an emphasised lead paragraph "
            "between the H1 and the first '##' (**Question.** / **Verdict:** / "
            "**Goal.**), or a first section that is the lead (## 一句話 / ## 兩句話 / ## 0.)"
        )

    if not LIMITS_PAT.search(text):
        fails.append(
            "no limits/honesty section (expected 不能說 / 未測 / 沒做的事 / 限制 / Caveats)"
        )

    if not REPRO_PAT.search(text):
        fails.append("no reproduction pointer (expected 重現 / Reproduce / 產物 / a scripts/ or data/ path)")

    plan = plan_sibling(path)
    if plan and plan not in text:
        fails.append(f"a pre-registration plan exists ({plan}) but the note never links it")

    for match in WIKILINK_PAT.finditer(text):
        warns.append(
            f"{match.group(0)} is a memory-file link — it dangles for any reader outside "
            "this machine; cite the note or the number in prose instead"
        )

    bare_refs = {m.group(0) for m in BARE_SECTION_PAT.finditer(text)}
    if bare_refs:
        warns.append(
            "cross-file section refs present ("
            + ", ".join(sorted(bare_refs)[:6])
            + "): confirm each is either a ref to THIS note's own section, or is "
            "spelled out inline — a reader without the plan open cannot resolve it"
        )

    # A coinage used ONLY as the parenthesised gloss right after its English term
    # ("within-class position（類內位置）") is exactly what the rule asks for.
    glossed = re.sub(r"[A-Za-z][A-Za-z0-9 \-]*（[^（）]{1,12}）", "", text)
    coined = {term: english for term, english in TRANSLATED_JARGON.items() if term in glossed}
    if coined:
        warns.append(
            "Chinese coinages used where the English term belongs ("
            + ", ".join(f"{term}→{english}" for term, english in sorted(coined.items())[:8])
            + "): technical terms stay in English with a Chinese gloss on first use only "
            "(SKILL.md 'Technical terms stay in English')"
        )

    for i, line in enumerate(text.splitlines(), 1):
        if PVALUE_PAT.search(line) and EQUIV_TRAP_PAT.search(line):
            warns.append(
                f"line {i}: a p-value and an equivalence claim in the same line — "
                "a non-significant result is 'undetermined', not 'no difference'"
            )

    return fails, warns


def diff_numbers(path: str) -> int:
    """Compare numeric tokens in git HEAD vs the working tree. 0 = identical."""
    try:
        old = subprocess.run(
            ["git", "show", f"HEAD:{path.replace(os.sep, '/')}"],
            capture_output=True, check=True,
        ).stdout.decode("utf-8")
    except subprocess.CalledProcessError:
        print(f"[numbers] {path}: not in git HEAD (new file) — nothing to compare")
        return 0

    def figures(text: str) -> set[str]:
        # Set, not multiset: a rewrite may legitimately cite the same figure a
        # different number of times. What must never happen is a figure vanishing.
        return set(NUMBER_PAT.findall(STRUCTURE_PAT.sub(" ", text)))

    a, b = figures(old), figures(read(path))
    dropped, added = sorted(a - b), sorted(b - a)

    if not dropped:
        print(f"[numbers] {path}: OK — all {len(a)} distinct figures from HEAD still present")
        if added:
            print(f"    (new figures introduced: {', '.join(added)} — each must be verified,"
                  " not derived from memory)")
        return 0

    print(f"[numbers] {path}: {len(dropped)} figure(s) present in HEAD but MISSING now")
    for tok in dropped:
        print(f"    dropped: {tok}")
    if added:
        print(f"    also new: {', '.join(added)}")
    print(
        "    A readability rewrite must preserve every measured figure. Check each\n"
        "    dropped token against the HEAD version before shipping."
    )
    return 1


def hook_mode() -> int:
    """PostToolUse hook: read the tool payload on stdin, lint notes/*.md, warn only.

    Always exits 0 — this never blocks a write. When there is something to say it
    emits additionalContext so the agent sees it and can fix the note in the same
    turn; otherwise it stays silent.
    """
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0

    tool_input = payload.get("tool_input") or {}
    tool_response = payload.get("tool_response") or {}
    path = tool_input.get("file_path") or tool_response.get("filePath") or ""
    norm = str(path).replace("\\", "/")
    if not (re.search(r"(^|/)notes/[^/]+\.md$", norm) and os.path.exists(path)):
        return 0

    fails, warns = check_file(path)
    if not fails and not warns:
        return 0

    lines = [f"note-convention check on {norm}:"]
    lines += [f"  FAIL {f}" for f in fails]
    lines += [f"  warn {w}" for w in warns]
    lines.append(
        "This is advisory, the write succeeded. Fix what is genuinely wrong before "
        "finishing; see .claude/skills/write-experiment-note/SKILL.md. If this is a "
        "rewrite, also run: check_note.py --numbers " + norm
    )
    json.dump(
        {
            "suppressOutput": True,
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "additionalContext": "\n".join(lines),
            },
        },
        sys.stdout,
    )
    return 0


def main(argv: list[str]) -> int:
    if "--hook" in argv[1:]:
        return hook_mode()

    args = [a for a in argv[1:] if a != "--numbers"]
    if not args:
        print(__doc__)
        return 2

    if "--numbers" in argv[1:]:
        return max(diff_numbers(p) for p in args)

    worst = 0
    for path in args:
        fails, warns = check_file(path)
        if fails:
            worst = 1
            print(f"FAIL {path}")
            for f in fails:
                print(f"  x {f}")
        else:
            print(f"pass {path}")
        for w in warns:
            print(f"  ! {w}")

    print(
        "\nStructural checks only. Register, jargon expansion, and whether each claim\n"
        "traces to a measured number are NOT machine-checkable — see SKILL.md."
    )
    return worst


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main(sys.argv))
