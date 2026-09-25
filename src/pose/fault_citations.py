"""The paper behind every fault a movement detector can report, read out of the detector source.

Each rule cites its literature in the ``citation=`` keyword of its ``build_detection(...)`` call. A real detection carries them to the studio's FaultCard; the movement
detail page's common-mistakes tab has no detection to read them from, so it reads
``frontend/src/lib/faultCitations.json`` instead, which ``main`` writes from this module. Both
therefore name the same paper for the same ``fault_id``.

Only the reference is exported. ``citation_support`` stays out: it is the English finding plus
the rule author's verification notes, and the page already carries a bilingual rendering of the
finding as each card's ``why``.

The source is walked as an AST rather than imported, for the reason
``tests/test_movement_mistakes_roster.py`` gives: the metadata lives in keyword arguments inside
rule bodies, and there is nothing to enumerate at runtime short of feeding each detector a video.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MOVEMENTS_DIR = REPO_ROOT / "src" / "pose" / "movements"
OUTPUT_JSON = REPO_ROOT / "frontend" / "src" / "lib" / "faultCitations.json"


def _string_constants(tree: ast.Module) -> dict[str, str]:
    """Module-level ``NAME = "..."`` assignments, so a keyword given as a constant resolves."""
    constants: dict[str, str] = {}
    for node in tree.body:
        target = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
        elif isinstance(node, ast.AnnAssign):
            target = node.target
        value = getattr(node, "value", None)
        if isinstance(target, ast.Name) and isinstance(value, ast.Constant) and isinstance(value.value, str):
            constants[target.id] = value.value
    return constants


def module_citations(source: str, label: str = "<source>") -> dict[str, str]:
    """``{fault_id: citation}`` for every ``build_detection`` call that cites a paper.

    A rule with no ``citation`` keyword, or an empty one, cites nothing and is left out. A keyword
    that is present but is not a string literal or module-level string constant raises instead of
    being skipped: skipping would drop that fault's paper from the page without anyone noticing.

    A fault built at more than one call site must cite one paper; two different ones raise, since
    the page shows one card per fault and could only pick one of them arbitrarily.
    """
    tree = ast.parse(source)
    constants = _string_constants(tree)

    def resolve(node: ast.expr, lineno: int, what: str) -> str:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name) and node.id in constants:
            return constants[node.id]
        raise ValueError(
            f"{label} line {lineno}: build_detection's {what} is not a string literal or a "
            "module-level string constant, so it cannot be read without running the detector."
        )

    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if name != "build_detection":
            continue
        keywords = {kw.arg: kw.value for kw in node.keywords}
        if "citation" not in keywords or "fault_id" not in keywords:
            continue
        citation = resolve(keywords["citation"], node.lineno, "citation")
        if not citation:
            continue
        fault_id = resolve(keywords["fault_id"], node.lineno, "fault_id")
        if out.get(fault_id, citation) != citation:
            raise ValueError(f"{label} line {node.lineno}: {fault_id} cites two different papers.")
        out[fault_id] = citation
    return out


def all_citations(movements_dir: Path = MOVEMENTS_DIR) -> dict[str, str]:
    """Every movement module's citations, merged and sorted by ``fault_id``."""
    merged: dict[str, str] = {}
    for path in sorted(movements_dir.glob("*.py")):
        merged.update(module_citations(path.read_text(encoding="utf-8"), path.name))
    return dict(sorted(merged.items()))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=OUTPUT_JSON)
    args = parser.parse_args(argv)
    args.output.write_text(
        json.dumps(all_citations(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output}")
