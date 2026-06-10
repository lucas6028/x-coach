"""FastAPI entry point for the x-coach web app.

Launch from the repository root so ``from src... import`` resolves:

    source .venv/bin/activate
    uvicorn backend.app.main:app --reload --port 8000
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app import config
from backend.app.routers import analyze, knowledge, videos

app = FastAPI(
    title="x-coach API",
    description="Explainable squat-coaching: pose perception + biomechanics rules + KG/RAG retrieval.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze.router)
app.include_router(videos.router)
app.include_router(knowledge.router)


@app.get("/api/health", tags=["meta"])
def health() -> dict:
    """Liveness check plus a quick view of which data stores are present."""
    return {
        "status": "ok",
        "stores": {
            "labeled_videos": config.VIDEOS_DIR.exists(),
            "detections": config.DETECTIONS_DIR.exists(),
            "kg_graph": config.KG_GRAPH_FILE.exists(),
            "rag_db": config.RAG_DB_DIR.exists(),
        },
    }


@app.on_event("startup")
def _startup() -> None:
    config.ensure_runtime_dirs()
