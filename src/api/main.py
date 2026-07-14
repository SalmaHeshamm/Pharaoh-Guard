"""
Application entry point. Run with:

    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

or via `docker compose up` (see docker-compose.yml at project root).
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import report
from config.settings import get_settings
from rag.knowledge_base import build_or_load_collection
from api.routes import chat, dispatch, explain, predict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Building/loading RAG knowledge base...")
    build_or_load_collection()
    logger.info("Startup complete.")
    yield
    logger.info("Shutting down.")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Pharaoh Guard",
        description="Agentic layer on top of the Risk_Score model — "
                     "explains, recommends, and dispatches responses to crowd/safety risk.",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(predict.router)
    app.include_router(explain.router)
    app.include_router(dispatch.router)
    app.include_router(report.router)
    app.include_router(chat.router)

    @app.get("/health", tags=["health"])
    def health() -> dict:
        return {"status": "ok"}

    return app


app = create_app()
