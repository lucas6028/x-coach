"""FastAPI entry point for the x-coach web app.

Launch from the repository root so ``from src... import`` resolves:

    source .venv/bin/activate
    uvicorn backend.app.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app import config
from backend.app.routers import (
    admin,
    analyses,
    analyze,
    auth_line,
    chat,
    conversations,
    knowledge,
    line_webhook,
    movements,
    videos,
)
from backend.app.settings import chat_models, default_chat_model, get_settings

logger = logging.getLogger(__name__)


def _log_storage_backend() -> None:
    """Announce which object store this process selected, once, at startup.

    ``storage_configured`` needs all four ``R2_*`` settings, and pydantic-settings ignores an
    unknown env var silently — so a single typo in a deploy environment flips the whole app to
    ``LocalObjectStore`` on an ephemeral disk AND leaves the unauthenticated dev endpoint
    ``GET /api/local-object/{key}`` live. That is a serious misconfiguration with no other
    symptom until uploads start disappearing, so it gets a WARNING in the deploy log (and a
    ``storage_configured`` field on ``/api/health``).

    LOGGING ONLY. The startup hook this replaces created runtime directories; object storage
    deliberately has no directories to create, and nothing here may resurrect that.
    """
    settings = get_settings()
    if getattr(settings, "storage_configured", False):
        logger.info(
            "Object storage: Cloudflare R2 (bucket=%s).", getattr(settings, "r2_bucket", "?")
        )
    else:
        logger.warning(
            "Object storage: LOCAL FILESYSTEM. R2 is not fully configured (needs R2_ACCOUNT_ID, "
            "R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY and R2_BUCKET) — uploads will not survive a "
            "redeploy, and the unauthenticated dev endpoint GET /api/local-object/{key} is live. "
            "Expected in development and CI; in production it means a misconfigured environment."
        )


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Startup/shutdown hooks. Startup logs the selected storage backend; nothing else."""
    _log_storage_backend()
    yield


app = FastAPI(
    lifespan=_lifespan,
    title="x-coach API",
    description="Explainable movement coaching: pose perception + biomechanics rules + KG/RAG retrieval over a 16-movement graph (video analysis covers Squat, Push-up and Overhead Press).",
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
app.include_router(analyses.router)
app.include_router(videos.router)
app.include_router(knowledge.router)
app.include_router(movements.router)
app.include_router(chat.router)
app.include_router(conversations.router)
app.include_router(admin.router)
app.include_router(auth_line.router)
app.include_router(line_webhook.router)


@app.get("/api/health", tags=["meta"])
def health() -> dict:
    """Liveness check plus a quick view of which data stores are present."""
    settings = get_settings()
    return {
        "status": "ok",
        "auth_configured": settings.auth_configured,
        "chat_configured": settings.chat_configured,
        # Whether the in-LIFF silent login (POST /api/auth/line) is available, so the
        # frontend only auto-attempts the exchange when the bridge is actually configured.
        # ``getattr`` default keeps this robust when a test patches ``get_settings`` to a
        # lightweight stand-in without the property (matching settings._allowed_base_hosts).
        "line_login_configured": bool(getattr(settings, "line_login_configured", False)),
        # False means uploads are going to an ephemeral local disk and the unauthenticated
        # ``/api/local-object`` dev endpoint is live — fine in development, a misconfiguration in
        # production. Same ``getattr`` guard as above, for the same reason.
        "storage_configured": bool(getattr(settings, "storage_configured", False)),
        # The Settings picker is server-driven: the selectable models + which is the default both
        # come from here (env-configurable), so the frontend never hard-codes the list.
        "chat_models": chat_models(),
        "chat_default": default_chat_model(),
        "stores": {
            "labeled_videos": config.VIDEOS_DIR.exists(),
            "detections": config.DETECTIONS_DIR.exists(),
            "kg_graph": config.KG_GRAPH_FILE.exists(),
            "rag_db": config.RAG_DB_DIR.exists(),
        },
    }
