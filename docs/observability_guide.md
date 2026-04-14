# Руководство по наблюдаемости (Observability)

## 1. Стек и обоснование выбора

### Сравнение вариантов

#### Логирование

| Инструмент | Pros | Cons | Итог |
|---|---|---|---|
| **structlog (выбрано)** | Нативный JSON, contextvars binding, интеграция со stdlib | Дополнительная зависимость | ✅ |
| loguru | Красивый вывод, прост в использовании | Нет машиночитаемого JSON, слабая интеграция | ❌ |
| stdlib logging | Zero deps | Многословная конфигурация, нет structured context | ❌ |
| python-json-logger | JSON из stdlib | Меньше возможностей, чем structlog | ❌ |

#### Метрики

| Инструмент | Pros | Cons | Итог |
|---|---|---|---|
| **prometheus_client (выбрано)** | HTTP /metrics endpoint, Grafana-совместим, zero infrastructure | — | ✅ |
| statsd | Простой протокол | UDP fire-and-forget, нет локального scrape | ❌ |
| OpenMetrics | Расширение Prometheus | Лишний overhead для простого кейса | ❌ |
| InfluxDB client | Time-series DB | Требует запущенного сервиса | ❌ |

#### Трейсинг

| Инструмент | Pros | Cons | Итог |
|---|---|---|---|
| **LangSmith (выбрано)** | Уже установлен как транзитивная зависимость langchain, zero-config | Требует API ключ для облака | ✅ |
| OpenTelemetry SDK | Стандарт индустрии | 4+ пакета + collector + Jaeger/Tempo | Можно добавить позже |
| Langfuse | Open-source LangSmith | Требует self-hosted сервиса | Альтернатива |

#### Алертинг

| Инструмент | Pros | Cons | Итог |
|---|---|---|---|
| **Prometheus AlertManager (выбрано)** | Интеграция с Prometheus, гибкие правила | Требует конфигурации | ✅ via docker-compose |
| PagerDuty / Opsgenie | Production-grade | Избыточно для локального развёртывания | ❌ |

---

## 2. Метрики

### Доступные метрики на `:8000/metrics`

```
# LLM вызовы
culinary_llm_calls_total{agent_name="recipe_finder", status="success"} 5.0
culinary_llm_calls_total{agent_name="nutritionist", status="error"} 1.0

# Латентность LLM (гистограмма, секунды)
culinary_llm_latency_seconds_bucket{agent_name="supervisor", le="10.0"} 8.0
culinary_llm_latency_seconds_sum{agent_name="supervisor"} 45.2

# Токены
culinary_llm_tokens_total{agent_name="recipe_finder", token_type="input"} 1024.0
culinary_llm_tokens_total{agent_name="recipe_finder", token_type="output"} 256.0

# Инструменты
culinary_tool_calls_total{tool_name="find_recipe_by_ingredients", status="success"} 3.0
culinary_tool_latency_seconds_sum{tool_name="calculate_nutrition"} 0.05

# Маршрутизация
culinary_supervisor_routes_total{destination_agent="recipe_finder"} 5.0
culinary_supervisor_routes_total{destination_agent="nutritionist"} 3.0

# Сессии
culinary_active_sessions 2.0
culinary_session_messages_total 15.0
```

---

## 3. Настройка Prometheus + Grafana

### Локальный запуск с мониторингом

```bash
# Запустить с профилем мониторинга
docker-compose --profile monitoring up

# Доступ:
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000  (admin/admin)
# Метрики:   http://localhost:8000/metrics
```

### Полезные PromQL запросы

```promql
# Средняя латентность LLM по агентам (последние 5 минут)
rate(culinary_llm_latency_seconds_sum[5m])
/ rate(culinary_llm_latency_seconds_count[5m])

# Процент ошибок LLM
rate(culinary_llm_calls_total{status="error"}[5m])
/ rate(culinary_llm_calls_total[5m]) * 100

# Распределение маршрутизации
sum by (destination_agent) (culinary_supervisor_routes_total)

# Топ-5 инструментов по вызовам
topk(5, culinary_tool_calls_total)
```

---

## 4. Логи

### Формат JSON (production)

```json
{
  "event": "llm_end",
  "agent": "recipe_finder",
  "latency_s": 3.421,
  "input_tokens": 512,
  "output_tokens": 128,
  "log_level": "info",
  "logger": "observability",
  "timestamp": "2026-04-14T12:00:00.123456Z"
}
```

### Формат текст (development)

```
2026-04-14 12:00:00 [info     ] llm_end   agent=recipe_finder latency_s=3.421
2026-04-14 12:00:00 [info     ] tool_end  tool=find_recipe_by_ingredients latency_s=0.001
2026-04-14 12:00:00 [info     ] supervisor_route  destination=recipe_finder
```

### Переключение формата

```bash
# .env или env var:
LOG_FORMAT=json   # для production / Loki
LOG_FORMAT=text   # для локальной разработки
```

---

## 5. LangSmith трейсинг (опционально)

```bash
# Активировать в .env:
ENABLE_LANGSMITH=true
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__your_key_here
LANGSMITH_PROJECT=purpo-culinary
```

После включения каждый вызов `app.invoke()` автоматически трассируется на [smith.langchain.com](https://smith.langchain.com) с полным деревом вызовов агентов и инструментов.

**Без API-ключа:** трейсинг отключён, никаких ошибок не возникает.

---

## 6. Пример алертинга (prometheus.yml)

```yaml
# Добавить в prometheus.yml:
rule_files:
  - "alerts.yml"
```

```yaml
# alerts.yml
groups:
  - name: culinary_alerts
    rules:
      - alert: HighLLMLatency
        expr: rate(culinary_llm_latency_seconds_sum[5m])
              / rate(culinary_llm_latency_seconds_count[5m]) > 30
        for: 2m
        annotations:
          summary: "LLM latency > 30s for 2 minutes"

      - alert: HighErrorRate
        expr: rate(culinary_llm_calls_total{status="error"}[5m])
              / rate(culinary_llm_calls_total[5m]) > 0.1
        for: 1m
        annotations:
          summary: "LLM error rate > 10%"
```
