"""Knowledge-graph and RAG query endpoints (for the KG widget and search)."""

from __future__ import annotations

from fastapi import APIRouter, Query

from backend.app.services import knowledge

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])


@router.get("/graph")
def graph(query: str = Query(..., min_length=1), hops: int = 1) -> dict:
    return knowledge.graph_context(query, hops=hops)


@router.get("/rag")
def rag(query: str = Query(..., min_length=1), top_k: int = 5) -> dict:
    return knowledge.rag_snippets(query, top_k=top_k)
