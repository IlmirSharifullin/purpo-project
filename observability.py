"""
Observability module for the culinary multi-agent assistant.

Stack decisions:
  - Logging: structlog — native JSON output, contextvars binding, integrates
    with stdlib so LangChain's own log calls flow through.
    Rejected: loguru (no machine-parseable JSON), stdlib (verbose config).

  - Metrics: prometheus_client — HTTP scrape endpoint on /metrics, works
    with Grafana. No extra service needed for local use.
    Rejected: statsd (fire-and-forget UDP, no local scrape endpoint).

  - Tracing: LangSmith (langsmith package already installed as transitive dep
    of langchain). Enable with LANGCHAIN_TRACING_V2=true env var.
    Rejected: OpenTelemetry (4+ packages + Jaeger/Tempo collector needed).

  - Alerting: Prometheus AlertManager via docker-compose (see docker-compose.yml).

Metrics exposed:
  culinary_llm_calls_total{agent_name, status}
  culinary_llm_latency_seconds{agent_name}
  culinary_llm_tokens_total{agent_name, token_type}
  culinary_tool_calls_total{tool_name, status}
  culinary_tool_latency_seconds{tool_name}
  culinary_supervisor_routes_total{destination_agent}
  culinary_active_sessions
  culinary_session_messages_total
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Sequence
from uuid import UUID

import structlog
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from prometheus_client import Counter, Gauge, Histogram, start_http_server

from config import settings

# ---------------------------------------------------------------------------
# Structlog configuration
# ---------------------------------------------------------------------------


def configure_logging() -> None:
    """
    Configure structlog. Call once at application startup.
    Produces JSON logs (log_format=json) or colored console output (log_format=text).
    """
    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if settings.log_format == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Route stdlib logging through structlog so LangChain logs are captured
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
    )


logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

LLM_CALLS_TOTAL = Counter(
    "culinary_llm_calls_total",
    "Total LLM inference calls",
    ["agent_name", "status"],
)
LLM_LATENCY_SECONDS = Histogram(
    "culinary_llm_latency_seconds",
    "LLM inference latency in seconds",
    ["agent_name"],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)
LLM_TOKENS_TOTAL = Counter(
    "culinary_llm_tokens_total",
    "Total tokens processed",
    ["agent_name", "token_type"],
)
TOOL_CALLS_TOTAL = Counter(
    "culinary_tool_calls_total",
    "Total tool invocations",
    ["tool_name", "status"],
)
TOOL_LATENCY_SECONDS = Histogram(
    "culinary_tool_latency_seconds",
    "Tool execution latency in seconds",
    ["tool_name"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
)
SUPERVISOR_ROUTES_TOTAL = Counter(
    "culinary_supervisor_routes_total",
    "Supervisor routing decisions",
    ["destination_agent"],
)
ACTIVE_SESSIONS = Gauge(
    "culinary_active_sessions",
    "Number of active conversation sessions",
)
SESSION_MESSAGES_TOTAL = Counter(
    "culinary_session_messages_total",
    "Total user messages processed",
)


def start_metrics_server() -> None:
    """Start Prometheus metrics HTTP server in a background daemon thread."""
    t = threading.Thread(
        target=start_http_server,
        args=(settings.metrics_port,),
        daemon=True,
    )
    t.start()
    logger.info("metrics_server_started", port=settings.metrics_port)


# ---------------------------------------------------------------------------
# LangChain callback handler — automatic instrumentation
# ---------------------------------------------------------------------------


class ObservabilityCallbackHandler(BaseCallbackHandler):
    """
    LangChain BaseCallbackHandler that records Prometheus metrics and
    structured log events for every LLM call, tool call, and chain step
    in the multi-agent graph.

    Usage:
        handler = ObservabilityCallbackHandler()
        result = app.invoke(
            {"messages": [...]},
            config={**run_config, "callbacks": [handler]},
        )
    """

    def __init__(self) -> None:
        super().__init__()
        self._llm_start: dict[str, float] = {}
        self._tool_start: dict[str, float] = {}

    # --- LLM lifecycle ---

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._llm_start[str(run_id)] = time.monotonic()
        agent_name = kwargs.get("name") or serialized.get("name", "unknown")
        structlog.contextvars.bind_contextvars(agent=agent_name, run_id=str(run_id))
        logger.debug(
            "llm_start",
            agent=agent_name,
            prompt_chars=len(prompts[0]) if prompts else 0,
        )

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        elapsed = time.monotonic() - self._llm_start.pop(str(run_id), time.monotonic())
        agent_name = kwargs.get("name", "unknown")
        LLM_CALLS_TOTAL.labels(agent_name=agent_name, status="success").inc()
        LLM_LATENCY_SECONDS.labels(agent_name=agent_name).observe(elapsed)

        usage = (getattr(response, "llm_output", None) or {}).get("token_usage", {})
        input_tok = usage.get("prompt_tokens", 0)
        output_tok = usage.get("completion_tokens", 0)
        if input_tok:
            LLM_TOKENS_TOTAL.labels(agent_name=agent_name, token_type="input").inc(input_tok)
        if output_tok:
            LLM_TOKENS_TOTAL.labels(agent_name=agent_name, token_type="output").inc(output_tok)

        logger.info(
            "llm_end",
            agent=agent_name,
            latency_s=round(elapsed, 3),
            input_tokens=input_tok,
            output_tokens=output_tok,
        )

    def on_llm_error(
        self, error: Exception, *, run_id: UUID, **kwargs: Any
    ) -> None:
        agent_name = kwargs.get("name", "unknown")
        LLM_CALLS_TOTAL.labels(agent_name=agent_name, status="error").inc()
        logger.error("llm_error", agent=agent_name, error=str(error))

    # --- Tool lifecycle ---

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        tool_name = serialized.get("name", "unknown")
        self._tool_start[str(run_id)] = time.monotonic()
        logger.debug("tool_start", tool=tool_name, input=input_str[:200])

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        elapsed = time.monotonic() - self._tool_start.pop(str(run_id), time.monotonic())
        tool_name = kwargs.get("name", "unknown")
        TOOL_CALLS_TOTAL.labels(tool_name=tool_name, status="success").inc()
        TOOL_LATENCY_SECONDS.labels(tool_name=tool_name).observe(elapsed)
        logger.info("tool_end", tool=tool_name, latency_s=round(elapsed, 3))

    def on_tool_error(
        self, error: Exception, *, run_id: UUID, **kwargs: Any
    ) -> None:
        tool_name = kwargs.get("name", "unknown")
        TOOL_CALLS_TOTAL.labels(tool_name=tool_name, status="error").inc()
        logger.error("tool_error", tool=tool_name, error=str(error))

    # --- Chain (agent / supervisor) lifecycle ---

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        chain_name = (serialized or {}).get("name", "unknown")
        if chain_name == "supervisor":
            SESSION_MESSAGES_TOTAL.inc()
        logger.debug("chain_start", chain=chain_name)

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        if not isinstance(outputs, dict):
            return
        for msg in outputs.get("messages", []):
            for tc in getattr(msg, "tool_calls", []):
                dest = (tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", ""))
                if dest.startswith("transfer_to_"):
                    agent = dest.replace("transfer_to_", "")
                    SUPERVISOR_ROUTES_TOTAL.labels(destination_agent=agent).inc()
                    logger.info("supervisor_route", destination=agent)
