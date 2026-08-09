"""Every kg-mode rule's kg_query must resolve to a node, for every movement.

This is a corpus gate, not a spot check. A kg_query that resolves to nothing is silent:
the detection still renders, but the coach gets `likely causes: -, injury risks: -,
corrective cues: -` for it. Squat never exposed this because all four of its strings
resolve; Overhead Press shipped with three of five broken.

CI SCOPE -- read before trusting "this is enforced":
`test_every_kg_query_resolves`, the test that actually checks resolution, is skipped
whenever `data/kg/sports_kg_v3.graphml` is absent. That graph is gitignored
(`.gitignore: data/*`) and `.github/workflows/ci.yml` never builds or fetches it, so
this test ALWAYS SKIPS in CI -- it only runs locally or on a deploy target that has
built the graph. `test_every_module_is_covered` and `test_queries_were_actually_found`
have no such dependency and DO run (and enforce) in CI; they are this file's only CI
teeth today. Giving the resolution check CI teeth would require CI to build or fetch
`sports_kg_v3.graphml` as a step before this suite runs -- not done as of this commit.
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GRAPH_FILE = REPO_ROOT / "data" / "kg" / "sports_kg_v3.graphml"
MOVEMENTS_DIR = REPO_ROOT / "src" / "pose" / "movements"

# module filename -> canonical movement name used for KG scoping.
MODULE_MOVEMENTS = {
    "squat.py": "Squat",
    "pushup.py": "Push-up",
    "overhead_press.py": "Overhead Press",
    "lunge.py": "Lunge",
    "deadlift.py": "Deadlift",
    "row.py": "Row",
    "band_pull_apart.py": "Band Pull Apart",
    "bicep_curl.py": "Bicep Curl",
}


def _module_string_constants(tree: ast.Module) -> dict[str, str]:
    """Top-level `NAME = "literal"` assignments, so a `kg_query=SOME_NAME` reference can be
    resolved to the string it names. `lunge.py` deliberately passes its four `kg_query` values
    as module constants (`LUNGE_PAST_TOES_KG_QUERY`, etc. -- see that module's Step 0 docstring)
    rather than retyping the literal at each call site, unlike squat/push-up/OHP which inline
    the string directly. Only MODULE-LEVEL assignments are resolved; anything else (an
    imported name, a computed value) is left unresolved and simply drops out of the corpus,
    same as before this function existed."""
    constants: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not (isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                constants[target.id] = node.value.value
    return constants


def _kg_queries(module_path: Path) -> list[str]:
    """Every kg_query= value in a detector module (literal string OR a reference to a
    module-level string constant), paired with the retrieval_mode of the same build_detection
    call. Parsed from the AST rather than imported, so this gate does not depend on
    numpy/detector import side effects."""
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    constants = _module_string_constants(tree)
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        kwargs = {kw.arg: kw.value for kw in node.keywords if kw.arg}
        query = kwargs.get("kg_query")
        mode = kwargs.get("retrieval_mode")
        if isinstance(query, ast.Constant) and isinstance(query.value, str):
            value = query.value
        elif isinstance(query, ast.Name) and query.id in constants:
            value = constants[query.id]
        else:
            continue
        if isinstance(mode, ast.Constant) and mode.value != "kg":
            continue  # rag-mode rules query the vector DB, not the graph
        found.append(value)
    return found


class TestKgQueryCorpus(unittest.TestCase):
    def test_every_module_is_covered(self) -> None:
        """Guard against a new detector module being added and silently skipped by this gate."""
        present = {p.name for p in MOVEMENTS_DIR.glob("*.py")} - {
            "__init__.py", "base.py", "registry.py"
        }
        self.assertEqual(present, set(MODULE_MOVEMENTS))

    def test_queries_were_actually_found(self) -> None:
        """A parser that silently returns [] would make the resolution test vacuously pass."""
        for filename in MODULE_MOVEMENTS:
            with self.subTest(module=filename):
                self.assertGreater(len(_kg_queries(MOVEMENTS_DIR / filename)), 0)

    @unittest.skipUnless(
        GRAPH_FILE.exists(),
        "sports_kg_v3.graphml not built in this checkout -- this test is gitignored out of "
        "CI entirely (data/* is ignored, CI never builds the graph); it only runs locally or "
        "on a deploy target where the graph has been built. The other two tests in this file "
        "run in CI and do not depend on the graph.",
    )
    def test_every_kg_query_resolves(self) -> None:
        from src.knowledge.graph_retrieval import retrieve_graph_context

        unresolved: list[str] = []
        for filename, movement in MODULE_MOVEMENTS.items():
            for query in _kg_queries(MOVEMENTS_DIR / filename):
                ctx = retrieve_graph_context(
                    query, graph_file=GRAPH_FILE, hops=1, max_seeds=3, movement=movement
                )
                if not ctx["matched_nodes"]:
                    unresolved.append(f"{movement}: {query!r}")
        self.assertEqual(unresolved, [], f"kg_query strings resolving to nothing: {unresolved}")
