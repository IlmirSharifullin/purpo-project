# Кулинарный мультиагентный ассистент

Локальный AI-помощник на базе LangGraph Supervisor + Ollama. Четыре специализированных агента отвечают на кулинарные запросы: рецепты, КБЖУ, список покупок и советы по готовке. Есть CLI-интерфейс и веб-чат.

## Возможности

- **Рецепты** — поиск блюд по имеющимся продуктам, пошаговые инструкции
- **КБЖУ** — расчёт калорий, белков, жиров и углеводов по граммам; коррекция рациона под цель
- **Список покупок** — недельный список по плану питания с оценкой бюджета в рублях
- **Советы по готовке** — замена ингредиентов, время и температура приготовления
- **Память** — контекст диалога сохраняется в SQLite, разные сессии через `thread_id`
- **Веб-интерфейс** — браузерный чат с историей сессий и рендерингом Markdown
- **Observability** — структурированные логи (structlog), метрики Prometheus, опциональный LangSmith

## Архитектура

```
Пользователь (CLI / Web)
        │
        ▼
  Супервайзер (LangGraph)
  ┌─────┬──────────┬──────────┬─────────────┐
  ▼     ▼          ▼          ▼             ▼
recipe  nutritionist  grocery  cooking
finder              list     coach
  │          │          │          │
tools      tools      tools      tools
        │
  SqliteSaver (data/checkpoints.db)
        │
  Prometheus metrics + structlog
```

Паттерн **Supervisor**: центральный агент анализирует запрос и делегирует нужному специалисту. При комбинированных запросах (рецепт + КБЖУ) агенты вызываются последовательно.

## Стек

| Компонент | Технология |
|---|---|
| LLM | Ollama (`qwen3.5:4b`, локально) |
| Мультиагент | LangGraph + `langgraph-supervisor` |
| Память | SQLite Checkpointer (`langgraph-checkpoint-sqlite`) |
| Веб-API | FastAPI + uvicorn |
| Логи | structlog |
| Метрики | prometheus-client |
| Конфигурация | pydantic-settings (`.env`) |
| Окружение | uv + Docker Compose |

## Быстрый старт

### Локально (CLI)

```bash
# Установить зависимости
uv sync

# Запустить Ollama и скачать модель
ollama pull qwen3.5:4b

# Запустить интерактивный REPL
uv run python main.py
```

### Веб-интерфейс

```bash
uv run python web/api.py
# Открыть http://localhost:8080
```

### Docker

```bash
# Только веб-сервис
docker-compose up --build web

# Веб + мониторинг (Prometheus + Grafana)
docker-compose --profile monitoring up --build web prometheus grafana
```

## Структура проекта

```
purpo-project/
├── main.py                 # CLI REPL + LangGraph граф
├── memory.py               # SQLite checkpointer + InMemoryStore
├── observability.py        # structlog + Prometheus callback
├── config.py               # Централизованная конфигурация (pydantic-settings)
├── evals.py                # Фреймворк оценки качества (9 тест-кейсов)
│
├── skills/                 # Системные промпты агентов (*.md)
│   ├── recipe_finder.md
│   ├── nutritionist.md
│   ├── grocery_list.md
│   └── cooking_coach.md
│
├── web/                    # Веб-интерфейс (самостоятельный модуль)
│   ├── api.py              # FastAPI приложение
│   └── static/
│       └── index.html      # Чат UI (vanilla JS, без сборки)
│
├── docs/                   # Проектная документация
│   ├── architecture.md
│   ├── TZ.md
│   ├── memory_design.md
│   ├── evals_guide.md
│   └── observability_guide.md
│
├── Dockerfile
├── docker-compose.yml
├── prometheus.yml
└── pyproject.toml
```

## Конфигурация

Настройки через переменные окружения или файл `.env`:

```env
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen3.5:4b
OLLAMA_TEMPERATURE=0.1
OLLAMA_NUM_CTX=4096

SQLITE_DB_PATH=data/checkpoints.db

LOG_LEVEL=INFO
LOG_FORMAT=text          # или json

METRICS_PORT=8000
API_HOST=0.0.0.0
API_PORT=8080

# Опционально: LangSmith трейсинг
ENABLE_LANGSMITH=false
LANGSMITH_PROJECT=purpo-culinary
```

## Оценка качества

```bash
# Запустить все 9 тест-кейсов
python evals.py

# Один кейс
python evals.py --case routing-05

# Отладка структуры сообщений
python evals.py --case routing-01 --debug
```

Метрики оценки: `routing_score` (40%) + `tool_use_score` (40%) + `content_score` (20%). Тест проходит при `composite_score ≥ 0.6`. Отчёт сохраняется в `evals_results/`.

## Мониторинг

| URL | Что |
|---|---|
| `http://localhost:8000/metrics` | Prometheus scrape (CLI) |
| `http://localhost:8001/metrics` | Prometheus scrape (Docker web) |
| `http://localhost:9090` | Prometheus UI |
| `http://localhost:3000` | Grafana (admin / admin) |

Доступные метрики: `culinary_llm_calls_total`, `culinary_llm_latency_seconds`, `culinary_tool_calls_total`, `culinary_supervisor_routes_total`, `culinary_session_messages_total`.

## Изменение поведения агентов

Поведение каждого агента определяется его `skills/*.md` файлом — он передаётся напрямую как системный промпт. Для изменения логики достаточно отредактировать `.md` файл, не трогая код.
