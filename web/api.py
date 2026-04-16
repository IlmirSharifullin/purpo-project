"""
FastAPI web interface for the culinary multi-agent assistant.

Run:
    python web/api.py           # from project root
    open http://localhost:8080

Endpoints:
    POST /chat      — send message, get reply
    POST /sessions  — create new session, returns thread_id
    GET  /health    — health check
    GET  /          — redirect to chat UI
    GET  /docs      — Swagger UI
"""

import sys
from pathlib import Path

# Add project root to sys.path so we can import main, memory, etc.
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import settings
from main import _get_last_assistant_message
from main import app as langgraph_app
from memory import build_run_config, new_thread_id
from observability import (
    ObservabilityCallbackHandler,
    configure_logging,
    start_metrics_server,
)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app_api = FastAPI(
    title="Кулинарный ассистент",
    description="Мультиагентный AI-помощник: рецепты, КБЖУ, покупки, готовка",
    version="1.0.0",
)

app_api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app_api.mount(
    "/static",
    StaticFiles(directory=Path(__file__).parent / "static"),
    name="static",
)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------


@app_api.on_event("startup")
async def startup() -> None:
    configure_logging()
    start_metrics_server()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None  # None → auto-generate


class ChatResponse(BaseModel):
    reply: str
    thread_id: str


class SessionResponse(BaseModel):
    thread_id: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _invoke_with_obs(message: str, tid: str) -> dict:
    """Run LangGraph synchronously (called via asyncio.to_thread)."""
    handler = ObservabilityCallbackHandler()
    run_cfg = build_run_config(thread_id=tid)
    return langgraph_app.invoke(
        {"messages": [{"role": "user", "content": message}]},
        config={**run_cfg, "callbacks": [handler]},
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app_api.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse(url="/static/index.html")


@app_api.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """Send a message to the culinary assistant and get a reply."""
    tid = req.thread_id or new_thread_id()
    result = await asyncio.to_thread(_invoke_with_obs, req.message, tid)
    reply = _get_last_assistant_message(result)
    return ChatResponse(reply=reply, thread_id=tid)


@app_api.post("/sessions", response_model=SessionResponse)
async def create_session() -> SessionResponse:
    """Create a new conversation session and return its thread_id."""
    return SessionResponse(thread_id=new_thread_id())


@app_api.get("/health")
async def health() -> dict:
    return {"status": "ok", "model": settings.ollama_model}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "api:app_api",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
        app_dir=str(Path(__file__).parent),
    )
