"""Knowledge-graph and RAG query endpoints (for the KG widget and search)."""

from __future__ import annotations

from fastapi import APIRouter, Query

from backend.app import settings
from backend.app.services import knowledge

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])

# These endpoints are public. ``hops`` (KG traversal depth) and ``top_k`` (RAG result count) feed
# directly into retrieval work, so they are bounded to sane ranges — an unbounded or negative
# value is rejected (422) rather than driving needless/expensive traversal or an invalid slice.
# A caller may still pass these query params to override; when omitted, the *default* comes from the
# admin-tunable getters so an operator can retune retrieval breadth without a redeploy.
@router.get("/graph")
def graph(
    query: str = Query(..., min_length=1),
    hops: int | None = Query(None, ge=1, le=3),
    movement: str | None = Query(None),
) -> dict:
    resolved_hops = hops if hops is not None else settings.kg_hops_default()
    return knowledge.graph_context(
        query, hops=resolved_hops, max_seeds=settings.kg_seeds_default(), movement=movement
    )


@router.get("/rag")
def rag(query: str = Query(..., min_length=1), top_k: int | None = Query(None, ge=1, le=50)) -> dict:
    resolved_top_k = top_k if top_k is not None else settings.rag_top_k_default()
    return knowledge.rag_snippets(query, top_k=resolved_top_k)
