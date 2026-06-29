"""Knowledge-graph and RAG query endpoints (for the KG widget and search)."""

from __future__ import annotations

from fastapi import APIRouter, Query

from backend.app.services import knowledge

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])

# These endpoints are public. ``hops`` (KG traversal depth) and ``top_k`` (RAG result count) feed
# directly into retrieval work, so they are bounded to sane ranges — an unbounded or negative
# value is rejected (422) rather than driving needless/expensive traversal or an invalid slice.
@router.get("/graph")
def graph(query: str = Query(..., min_length=1), hops: int = Query(1, ge=1, le=3)) -> dict:
    return knowledge.graph_context(query, hops=hops)


@router.get("/rag")
def rag(query: str = Query(..., min_length=1), top_k: int = Query(5, ge=1, le=50)) -> dict:
    return knowledge.rag_snippets(query, top_k=top_k)
