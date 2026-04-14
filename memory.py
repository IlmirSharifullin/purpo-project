"""
Memory module for the culinary multi-agent assistant.

Two-layer architecture:
  1. Short-term (conversation): SqliteSaver — persists every LangGraph superstep
     as a checkpoint under a thread_id. Fallback: InMemorySaver.
  2. Long-term (user profile): InMemoryStore — stores dietary goals, restrictions,
     budget preferences across conversations.

Upgrade path:
  - Replace InMemoryStore with langgraph-checkpoint-postgres AsyncPostgresStore
    for persistent long-term storage (single import line change).
  - Replace SqliteSaver with AsyncSqliteSaver for async workflows.

Known limitations of this approach:
  - InMemoryStore loses long-term profiles on process restart.
  - SQLite is single-writer: concurrent sessions will serialize writes.
    Acceptable for local/single-user deployment.
"""

from __future__ import annotations

import uuid
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from config import settings

# ---------------------------------------------------------------------------
# Long-term store — user dietary preferences and goals
# ---------------------------------------------------------------------------

_long_term_store = InMemoryStore()


def get_user_profile(user_id: str) -> dict[str, Any]:
    """Retrieve stored user profile. Returns empty dict if not found."""
    item = _long_term_store.get(("users", user_id), "profile")
    if item is None:
        return {}
    return item.value


def update_user_profile(user_id: str, updates: dict[str, Any]) -> None:
    """Merge updates into the user's persistent profile."""
    current = get_user_profile(user_id)
    current.update(updates)
    current["_updated_at"] = datetime.utcnow().isoformat()
    _long_term_store.put(("users", user_id), "profile", current)


def get_store() -> InMemoryStore:
    """Return the global long-term store (inject into graph compile)."""
    return _long_term_store


# ---------------------------------------------------------------------------
# Short-term checkpointer — conversation state per thread_id
# ---------------------------------------------------------------------------


def build_checkpointer():
    """
    Returns a SqliteSaver for persistent conversation memory.
    Falls back to InMemorySaver if langgraph-checkpoint-sqlite is not installed.

    Usage:
        checkpointer = build_checkpointer()
        app = workflow.compile(checkpointer=checkpointer, store=get_store())
    """
    try:
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver  # type: ignore[import]

        db_path = Path(settings.sqlite_db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        return SqliteSaver(conn)
    except (ImportError, Exception):
        warnings.warn(
            "langgraph-checkpoint-sqlite not available. "
            "Using InMemorySaver (no persistence across restarts). "
            "Install with: uv add langgraph-checkpoint-sqlite",
            stacklevel=2,
        )
        return InMemorySaver()


# ---------------------------------------------------------------------------
# Thread and config helpers
# ---------------------------------------------------------------------------


def new_thread_id() -> str:
    """Generate a unique conversation thread ID."""
    return f"thread-{uuid.uuid4().hex[:8]}"


def build_run_config(
    thread_id: str | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    """
    Build the LangGraph RunnableConfig dict for a single invocation.

    Args:
        thread_id: Conversation thread. Uses settings.default_thread_id if None.
        user_id: Optional user identifier for profile lookup.

    Returns:
        dict suitable for passing as config= to app.invoke().
    """
    tid = thread_id or settings.default_thread_id
    cfg: dict[str, Any] = {"configurable": {"thread_id": tid}}
    if user_id:
        cfg["configurable"]["user_id"] = user_id
    return cfg


def trim_messages_to_window(messages: list, max_messages: int | None = None) -> list:
    """
    Keep only the last N messages to prevent context overflow with small LLMs.
    Always preserves the first (system) message if present.
    """
    limit = max_messages or settings.memory_window_size
    if len(messages) <= limit:
        return messages
    system = [m for m in messages[:1] if getattr(m, "type", None) == "system"]
    tail = messages[-(limit - len(system)):]
    return system + tail
