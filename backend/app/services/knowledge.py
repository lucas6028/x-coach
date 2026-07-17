"""Knowledge service: thin wrappers over the KG and RAG retrieval entry points in ``src/``."""

from __future__ import annotations

from typing import Any

from src.knowledge.graph_retrieval import retrieve_graph_context
from src.knowledge.rag_vector_db import query_vector_db

from backend.app import config


def graph_context(
    query: str, *, hops: int = 1, max_seeds: int = 5, movement: str | None = None
) -> dict[str, Any]:
    """Return the knowledge-graph subgraph + summaries for a fault/query string."""
    return retrieve_graph_context(
        query,
        graph_file=config.KG_GRAPH_FILE,
        hops=hops,
        max_seeds=max_seeds,
        movement=movement,
    )


def rag_snippets(query: str, *, top_k: int = 5) -> dict[str, Any]:
    """Return ranked RAG text snippets for a query string."""
    return {
        "query": query,
        "results": query_vector_db(query, db_dir=config.RAG_DB_DIR, top_k=top_k),
    }
