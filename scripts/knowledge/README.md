# Knowledge And RAG Scripts

Knowledge workflow entry points for adding documents, building/querying the local RAG index, and maintaining the multi-movement sports knowledge graph (`data/kg/sports_kg_v3.graphml`).

## Local RAG

```bash
# Add a document to the local RAG document area and metadata file.
python scripts/knowledge/add_document.py path/to/article.txt --type health_blog --ref "Blog: Top 10 Squat Tips"

# Rebuild the local vector database.
python scripts/knowledge/build_rag_vector_db.py build

# Query the local vector database.
python scripts/knowledge/query_rag.py query "knees inward correction"
```

## Knowledge Graph

```bash
# Extract knowledge graph data from a document.
python src/knowledge/extract_kg.py path/to/document.pdf

# Clean a graph with a canonical mapping file.
python scripts/knowledge/clean_kg.py \
  --graph-file data/kg/squat_kg_v2.graphml \
  --mapping-file data/kg/docs/squat_canonical_mapping_v1.json \
  --output-file data/kg/squat_kg_v2_cleaned.graphml

# Query graph content. (`audit_kg.py` still points at the legacy squat_kg_v2 graph, not v3.)
python scripts/knowledge/query_graph.py "knee valgus"
```

### Rebuilding `sports_kg_v3.graphml` from scratch

`data/kg/*.graphml` is **gitignored** (`.gitignore` excludes `data/*` wholesale), so the graph
is a build artifact and these scripts are the reproducible deliverable. Run from the repo root
with `.venv\Scripts\python.exe`.

⚠️ **Step 1's input is not in git either.** `migrate_to_v3.py` reads
`data/kg/squat_kg_v2.graphml`, which is ignored by the same rule — a fresh clone cannot start
this sequence. Copy `squat_kg_v2.graphml` (or the finished `sports_kg_v3.graphml`) from a
machine that has one.

The order below is **the order the current graph was actually built in**, recovered from the
`data/kg/*.bak` / `*.pre-*` snapshot timestamps; it is not a claim that each script hard-checks
its predecessor. Every one of them is idempotent or aborts rather than double-applying:

| # | Script | Adds |
|---|---|---|
| 1 | `migrate_to_v3.py` | v1/v2 → v3 scoped/shared schema |
| 2 | `reconcile_lunge_v3.py` | Lunge flagship |
| 3 | `reconcile_pushup_v3.py` | Push-up flagship |
| 4 | `reconcile_ohp_v3.py` | Overhead Press flagship |
| 5 | `reconcile_row_v3.py` | Row flagship |
| 6 | `stub_general_movements_v3.py` | the 11 general-movement stubs |
| 7 | `author_ohp_lockout_v3.py` | OHP `Incomplete Elbow Lockout` |
| 8 | `author_band_pull_apart_kg_v3.py` | Band Pull Apart `Trunk Extension Compensation` + the `Bent Elbows` edge |

Steps 7–8 patch a graph step 6 has already built: `stub_general_movements_v3.py` skips any
movement whose `Action` node exists, so editing its `STUB_SPEC` alone cannot fix an existing
graph. Where a later script duplicates a `STUB_SPEC` entry, both carry the same content on
purpose — its docstring says which.

**Deploying the result is a file upload, not a rebuild.** Docker bind-mounts `./data`
(`docker-compose.yml`) and Azure Files receives the local file via `az storage file upload`
(`docs/azure-deployment.md` §"灌入資料共用"), so re-running a script here only reaches
production once `data/kg/sports_kg_v3.graphml` is re-uploaded.

The data tests `tests/test_kg_ohp_lockout.py` and `tests/test_kg_band_pull_apart.py` assert
against this graph and **skip** when it is absent, so a fresh clone is green whether or not
the graph was built.
