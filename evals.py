"""
Evaluation framework for the culinary multi-agent assistant.

Two evaluation dimensions:
  1. LLM Quality — response correctness, keyword grounding, no hallucinated data.
  2. Agentic System — routing correctness (right agent called?), tool use
     (right tools invoked?), multi-agent completion.

Metrics per eval case:
  routing_score   — fraction of expected_agents actually called (0.0–1.0)
  tool_use_score  — fraction of expected_tools invoked (0.0–1.0)
  content_score   — keyword presence + no forbidden keywords + min length (0.0–1.0)
  composite_score — weighted: routing 40% + tool_use 40% + content 20%

Usage:
    python evals.py                  # run built-in suite against local Ollama
    python evals.py --case routing-01  # run single case
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from config import settings


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class EvalCase:
    """A single evaluation test case."""

    case_id: str
    description: str
    user_input: str
    expected_agents: list[str]
    expected_tools: list[str]
    expected_keywords: list[str]
    forbidden_keywords: list[str] = field(default_factory=list)
    min_response_length: int = 20
    thread_id: str | None = None


@dataclass
class EvalResult:
    """Result of running one EvalCase."""

    case_id: str
    passed: bool
    score: float
    latency_s: float
    agents_called: list[str]
    tools_called: list[str]
    final_response: str
    failures: list[str]
    raw_output: dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    routing_score: float = 0.0
    tool_use_score: float = 0.0
    content_score: float = 0.0


@dataclass
class EvalReport:
    """Aggregate report across all test cases."""

    run_id: str
    timestamp: str
    total_cases: int
    passed: int
    failed: int
    pass_rate: float
    avg_score: float
    avg_latency_s: float
    results: list[EvalResult]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    def print_summary(self) -> None:
        print(f"\n{'='*60}")
        print(f"EVAL REPORT  {self.timestamp}")
        print(f"{'='*60}")
        print(f"Total cases : {self.total_cases}")
        print(f"Passed      : {self.passed}  ({self.pass_rate*100:.0f}%)")
        print(f"Failed      : {self.failed}")
        print(f"Avg score   : {self.avg_score:.2f}")
        print(f"Avg latency : {self.avg_latency_s:.1f}s")
        print(f"{'='*60}")
        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            print(
                f"  [{status}] {r.case_id:<20} score={r.score:.2f}  "
                f"latency={r.latency_s:.1f}s"
            )
            for f in r.failures:
                print(f"         ✗ {f}")
        print()


# ---------------------------------------------------------------------------
# Built-in test suite (9 cases)
# ---------------------------------------------------------------------------

EVAL_SUITE: list[EvalCase] = [
    EvalCase(
        case_id="routing-01",
        description="Запрос рецепта → recipe_finder",
        user_input="Что можно приготовить из гречки и грибов?",
        expected_agents=["recipe_finder"],
        expected_tools=["find_recipe_by_ingredients"],
        expected_keywords=["рецепт", "греч", "гриб"],
    ),
    EvalCase(
        case_id="routing-02",
        description="Запрос КБЖУ → nutritionist",
        user_input="Сколько калорий в 150 граммах гречки?",
        expected_agents=["nutritionist"],
        expected_tools=["calculate_nutrition"],
        expected_keywords=["ккал", "белк", "жир", "углевод"],
    ),
    EvalCase(
        case_id="routing-03",
        description="Список покупок → grocery_list",
        user_input="Составь список покупок на неделю: курица, гречка, яйца, творог",
        expected_agents=["grocery_list"],
        expected_tools=["create_weekly_grocery_list"],
        expected_keywords=["покупок", "мясо", "молочн"],
    ),
    EvalCase(
        case_id="routing-04",
        description="Замена ингредиента → cooking_coach",
        user_input="Чем можно заменить сливочное масло в рецепте?",
        expected_agents=["cooking_coach"],
        expected_tools=["find_ingredient_substitute"],
        expected_keywords=["замен", "масл"],
    ),
    EvalCase(
        case_id="routing-05",
        description="Мультиагент: рецепт + КБЖУ — воспроизводит баг из output.json",
        user_input=(
            "Что можно приготовить из курицы? "
            "И отдельно: посчитай КБЖУ для куриной грудки 200 граммов."
        ),
        expected_agents=["recipe_finder", "nutritionist"],
        expected_tools=["find_recipe_by_ingredients", "calculate_nutrition"],
        expected_keywords=["рецепт", "ккал", "белк"],
        forbidden_keywords=["формула КБЖУ ="],
    ),
    EvalCase(
        case_id="routing-06",
        description="Время готовки → cooking_coach",
        user_input="Сколько времени варить куриную грудку?",
        expected_agents=["cooking_coach"],
        expected_tools=["get_cooking_time"],
        expected_keywords=["мин", "вар"],
    ),
    EvalCase(
        case_id="budget-01",
        description="Оценка бюджета → grocery_list",
        user_input="Сколько примерно стоит стандартный список продуктов на неделю?",
        expected_agents=["grocery_list"],
        expected_tools=["estimate_budget"],
        expected_keywords=["руб", "итого"],
    ),
    EvalCase(
        case_id="diet-01",
        description="Коррекция рациона под цель похудения → nutritionist",
        user_input="Я хочу похудеть. Сейчас ем около 2500 калорий в день. Что посоветуешь?",
        expected_agents=["nutritionist"],
        expected_tools=["adjust_diet_for_goal"],
        expected_keywords=["похудение", "дефицит", "ккал"],
    ),
    EvalCase(
        case_id="grounding-01",
        description="Ответ должен содержать точное число из инструмента, не галлюцинацию",
        user_input="Сколько белка в 100 граммах творога?",
        expected_agents=["nutritionist"],
        expected_tools=["calculate_nutrition"],
        expected_keywords=["18"],  # exact value from the tool DB
        forbidden_keywords=["не знаю", "примерно", "около 10"],
    ),
]


# ---------------------------------------------------------------------------
# Metric extractors
# ---------------------------------------------------------------------------


def _extract_agents_called(raw_output: dict) -> list[str]:
    agents: list[str] = []
    for msg in raw_output.get("messages", []):
        name = getattr(msg, "name", None)
        if not name and isinstance(msg, dict):
            name = msg.get("name")
        if not name or name in agents:
            continue
        if name in ("supervisor",):
            continue
        content = getattr(msg, "content", "") or (
            msg.get("content", "") if isinstance(msg, dict) else ""
        )
        if "Successfully transferred" not in str(content):
            agents.append(name)
    return agents


def _extract_tools_called(raw_output: dict) -> list[str]:
    tools: list[str] = []
    for msg in raw_output.get("messages", []):
        if "Tool" in type(msg).__name__:
            name = getattr(msg, "name", None)
            if name and not name.startswith("transfer") and name not in tools:
                tools.append(name)
    return tools


def _extract_final_response(raw_output: dict) -> str:
    messages = raw_output.get("messages", [])

    # Primary: last AIMessage with non-empty text content
    for msg in reversed(messages):
        content = getattr(msg, "content", "") or (
            msg.get("content", "") if isinstance(msg, dict) else ""
        )
        msg_type = type(msg).__name__
        if content and "Tool" not in msg_type and "transfer" not in str(content).lower():
            return str(content)

    # Fallback: last ToolMessage that is a real tool result (not a transfer confirmation)
    for msg in reversed(messages):
        msg_type = type(msg).__name__
        if "Tool" not in msg_type:
            continue
        name = getattr(msg, "name", "") or (msg.get("name", "") if isinstance(msg, dict) else "")
        content = getattr(msg, "content", "") or (
            msg.get("content", "") if isinstance(msg, dict) else ""
        )
        if (
            content
            and "transfer" not in str(name).lower()
            and "Successfully transferred" not in str(content)
        ):
            return str(content)

    return ""


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


def _score_result(
    case: EvalCase,
    agents_called: list[str],
    tools_called: list[str],
    final_response: str,
) -> tuple[float, float, float, list[str]]:
    """Returns (routing_score, tool_use_score, content_score, failures)."""
    failures: list[str] = []

    # Routing score
    if case.expected_agents:
        found = sum(
            1 for a in case.expected_agents
            if any(a in called for called in agents_called)
        )
        routing = found / len(case.expected_agents)
        if routing < 1.0:
            missing = [a for a in case.expected_agents if not any(a in c for c in agents_called)]
            failures.append(f"Агенты не вызваны: {missing}")
    else:
        routing = 1.0

    # Tool use score
    if case.expected_tools:
        found = sum(
            1 for t in case.expected_tools
            if any(t in called for called in tools_called)
        )
        tool_use = found / len(case.expected_tools)
        if tool_use < 1.0:
            missing = [t for t in case.expected_tools if not any(t in c for c in tools_called)]
            failures.append(f"Инструменты не вызваны: {missing}")
    else:
        tool_use = 1.0

    # Content score
    resp_lower = final_response.lower()
    keyword_hits = sum(1 for kw in case.expected_keywords if kw.lower() in resp_lower)
    keyword_score = keyword_hits / len(case.expected_keywords) if case.expected_keywords else 1.0

    forbidden_hits = [kw for kw in case.forbidden_keywords if kw.lower() in resp_lower]
    forbidden_penalty = 0.0 if not forbidden_hits else 1.0
    if forbidden_hits:
        failures.append(f"Запрещённые слова в ответе: {forbidden_hits}")

    length_ok = len(final_response) >= case.min_response_length
    if not length_ok:
        failures.append(f"Ответ слишком короткий: {len(final_response)} < {case.min_response_length}")

    content = keyword_score * (1.0 - forbidden_penalty) * (1.0 if length_ok else 0.5)

    return routing, tool_use, content, failures


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_eval(
    app: Any,
    case: EvalCase,
    build_config: Callable[[str], dict] | None = None,
) -> EvalResult:
    """
    Run a single EvalCase against the compiled LangGraph app.

    Args:
        app: Compiled LangGraph app (workflow.compile()).
        case: The test case to execute.
        build_config: Optional callable(thread_id) -> LangGraph config dict.
    """
    thread_id = case.thread_id or f"eval-{uuid.uuid4().hex[:8]}"
    if build_config is None:
        run_cfg: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    else:
        run_cfg = build_config(thread_id)

    raw_output: dict[str, Any] = {}
    t0 = time.monotonic()
    try:
        raw_output = app.invoke(
            {"messages": [{"role": "user", "content": case.user_input}]},
            config=run_cfg,
        )
    except Exception as exc:
        raw_output = {"messages": [], "_error": str(exc)}

    latency = time.monotonic() - t0

    agents_called = _extract_agents_called(raw_output)
    tools_called = _extract_tools_called(raw_output)
    final_response = _extract_final_response(raw_output)

    routing, tool_use, content, failures = _score_result(
        case, agents_called, tools_called, final_response
    )
    composite = routing * 0.4 + tool_use * 0.4 + content * 0.2
    passed = composite >= 0.6 and len(failures) == 0

    return EvalResult(
        case_id=case.case_id,
        passed=passed,
        score=round(composite, 3),
        latency_s=round(latency, 2),
        agents_called=agents_called,
        tools_called=tools_called,
        final_response=final_response[:500],
        failures=failures,
        raw_output={},  # omit to keep report small
        routing_score=round(routing, 3),
        tool_use_score=round(tool_use, 3),
        content_score=round(content, 3),
    )


def run_eval_suite(
    app: Any,
    suite: list[EvalCase] | None = None,
    build_config: Callable[[str], dict] | None = None,
) -> EvalReport:
    """Run all cases in the suite and return an aggregate EvalReport."""
    if suite is None:
        suite = EVAL_SUITE

    results: list[EvalResult] = []
    for case in suite:
        print(f"  Running {case.case_id} …", end=" ", flush=True)
        result = run_eval(app, case, build_config)
        results.append(result)
        print("PASS" if result.passed else f"FAIL (score={result.score:.2f})")

    passed = sum(1 for r in results if r.passed)
    avg_score = sum(r.score for r in results) / len(results) if results else 0.0
    avg_latency = sum(r.latency_s for r in results) / len(results) if results else 0.0

    return EvalReport(
        run_id=uuid.uuid4().hex[:8],
        timestamp=datetime.utcnow().isoformat(),
        total_cases=len(results),
        passed=passed,
        failed=len(results) - passed,
        pass_rate=passed / len(results) if results else 0.0,
        avg_score=round(avg_score, 3),
        avg_latency_s=round(avg_latency, 2),
        results=results,
    )


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def _build_app():
    """Import and return the compiled app from main.py."""
    import main  # noqa: PLC0415
    return main.app


def _debug_messages(raw_output: dict) -> None:
    """Print the type and content snippet of every message in the output."""
    messages = raw_output.get("messages", [])
    print(f"\n--- DEBUG: {len(messages)} messages ---")
    for i, msg in enumerate(messages):
        msg_type = type(msg).__name__
        name = getattr(msg, "name", None) or (msg.get("name") if isinstance(msg, dict) else None)
        content = getattr(msg, "content", "") or (msg.get("content", "") if isinstance(msg, dict) else "")
        snippet = str(content)[:120].replace("\n", "↵")
        print(f"  [{i}] {msg_type:20s}  name={name!r:25s}  content={snippet!r}")
    print("--- END DEBUG ---\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run culinary assistant evaluations")
    parser.add_argument("--case", help="Run only this case ID (e.g. routing-01)")
    parser.add_argument("--debug", action="store_true", help="Print raw message structure for each case")
    args = parser.parse_args()

    print("Загрузка приложения …")
    app = _build_app()

    suite = EVAL_SUITE
    if args.case:
        suite = [c for c in EVAL_SUITE if c.case_id == args.case]
        if not suite:
            print(f"Кейс '{args.case}' не найден. Доступные: {[c.case_id for c in EVAL_SUITE]}")
            raise SystemExit(1)

    if args.debug:
        # Run one case and dump the raw message structure, then exit
        case = suite[0]
        print(f"Запуск {case.case_id} в режиме отладки …")
        run_cfg: dict[str, Any] = {"configurable": {"thread_id": f"eval-debug-{uuid.uuid4().hex[:6]}"}}
        try:
            raw = app.invoke(
                {"messages": [{"role": "user", "content": case.user_input}]},
                config=run_cfg,
            )
        except Exception as exc:
            raw = {"messages": [], "_error": str(exc)}
            print(f"ОШИБКА: {exc}")
        _debug_messages(raw)
        raise SystemExit(0)

    print(f"Запуск {len(suite)} тест-кейсов …\n")
    report = run_eval_suite(app, suite)
    report.print_summary()

    out_dir = Path(settings.evals_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"report_{report.run_id}.json"
    out_path.write_text(report.to_json(), encoding="utf-8")
    print(f"Отчёт сохранён: {out_path}")
