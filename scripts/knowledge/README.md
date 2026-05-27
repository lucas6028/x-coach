# Knowledge And RAG Scripts

Knowledge workflow entry points for adding documents, building/querying the local RAG index, and maintaining the squat knowledge graph.

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

# Audit or query graph content.
python scripts/knowledge/audit_kg.py
python scripts/knowledge/query_graph.py "knee valgus"
```
