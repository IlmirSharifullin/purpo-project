# Руководство по оценке системы (Evals)

## 1. Что оцениваем

Система имеет два уровня оценки:

### Уровень 1: Качество LLM-ответа
- **Grounding** — ответ содержит точные данные из инструментов, а не галлюцинации
- **Completeness** — все ключевые слова/понятия присутствуют в ответе
- **Safety** — запрещённые паттерны (выдуманные данные) отсутствуют
- **Length** — ответ не слишком короткий (минимум 20 символов)

### Уровень 2: Производительность агентной системы
- **Routing accuracy** — правильный агент был вызван
- **Tool use** — правильные инструменты были вызваны (а не ответ из памяти модели)
- **Multi-agent completion** — при составном запросе все части получили ответ

---

## 2. Метрики

| Метрика | Формула | Вес в итоговом score |
|---|---|---|
| `routing_score` | (вызванные ожидаемые агенты) / (всего ожидаемых) | 40% |
| `tool_use_score` | (вызванные ожидаемые инструменты) / (всего ожидаемых) | 40% |
| `content_score` | keyword_hit_rate × (1 − forbidden_penalty) × length_ok | 20% |
| `composite_score` | routing×0.4 + tool_use×0.4 + content×0.2 | — |

**Порог прохождения:** `composite_score ≥ 0.6` и `len(failures) == 0`

---

## 3. Встроенные тест-кейсы

| ID | Описание | Что проверяет |
|---|---|---|
| routing-01 | Запрос рецепта | recipe_finder + find_recipe_by_ingredients |
| routing-02 | Запрос КБЖУ | nutritionist + calculate_nutrition |
| routing-03 | Список покупок | grocery_list + create_weekly_grocery_list |
| routing-04 | Замена ингредиента | cooking_coach + find_ingredient_substitute |
| routing-05 | **Мультиагент** (баг из output.json) | Оба агента вызваны для составного запроса |
| routing-06 | Время готовки | cooking_coach + get_cooking_time |
| budget-01 | Оценка бюджета | grocery_list + estimate_budget |
| diet-01 | Коррекция рациона | nutritionist + adjust_diet_for_goal |
| grounding-01 | Проверка грандинга | Точное значение из инструмента (18г белка в твороге) |

---

## 4. Запуск оценки

### Полный suite
```bash
# Запуск всех 9 тест-кейсов
python evals.py
```

### Один кейс
```bash
python evals.py --case routing-05
```

### Программный запуск
```python
from evals import run_eval_suite, EVAL_SUITE
import main

report = run_eval_suite(main.app)
report.print_summary()
print(report.to_json())
```

### Вывод
```
============================================================
EVAL REPORT  2026-04-14T12:00:00
============================================================
Total cases : 9
Passed      : 7  (78%)
Failed      : 2
Avg score   : 0.72
Avg latency : 23.4s
============================================================
  [PASS] routing-01           score=1.00  latency=18.2s
  [PASS] routing-02           score=1.00  latency=21.5s
  [FAIL] routing-05           score=0.50  latency=35.1s
         ✗ Агенты не вызваны: ['nutritionist']
  ...
```

---

## 5. Добавление собственных тест-кейсов

```python
from evals import EvalCase, EVAL_SUITE

my_case = EvalCase(
    case_id="my-test-01",
    description="Мой тест: запрос рецепта с рыбой",
    user_input="Что приготовить из лосося?",
    expected_agents=["recipe_finder"],
    expected_tools=["find_recipe_by_ingredients"],
    expected_keywords=["лосось", "рецепт"],
    min_response_length=50,
)

# Добавить в suite:
EVAL_SUITE.append(my_case)
```

---

## 6. Интерпретация результатов

| composite_score | Интерпретация |
|---|---|
| 0.9 – 1.0 | Отлично: агент работает как ожидается |
| 0.7 – 0.9 | Хорошо: незначительные отклонения |
| 0.5 – 0.7 | Предупреждение: проверить промпты |
| < 0.5 | Проблема: агент не маршрутизируется или не использует инструменты |

**Типичные причины отказа:**
- `routing_score < 1.0` → Супервайзер ответил сам, не делегировал → Улучшить системный промпт
- `tool_use_score < 1.0` → Агент ответил из памяти модели → Добавить ОБЯЗАТЕЛЬНО в промпт агента
- `content_score < 1.0` → Ответ неполный или содержит запрещённые слова → Проверить инструмент

---

## 7. Хранение отчётов

Отчёты сохраняются в `evals_results/report_<run_id>.json`.

Структура файла:
```json
{
  "run_id": "a3f8c2d1",
  "timestamp": "2026-04-14T12:00:00",
  "total_cases": 9,
  "passed": 7,
  "pass_rate": 0.78,
  "avg_score": 0.72,
  "results": [...]
}
```
