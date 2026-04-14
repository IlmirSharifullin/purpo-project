from langchain_ollama import ChatOllama

model = ChatOllama(model="qwen2.5:3b")  # проверь точное имя через ollama list

from langchain.tools import tool

@tool
def add(a: float, b: float) -> float:
    """Сложить два числа."""
    return a + b

@tool
def multiply(a: float, b: float) -> float:
    """Умножить два числа."""
    return a * b

@tool
def web_search(query: str) -> str:
    """Поиск информации в интернете."""
    return (
        "Сегодня в Казани солнечно, температура около 25 градусов. В мире продолжаются обсуждения по поводу изменения климата и его влияния на окружающую среду."
    )


from langchain.agents import create_agent

math_agent = create_agent(
    model=model,
    tools=[add, multiply],
    name="math_expert",
    system_prompt=(
        "Вы - эксперт по математике. "
        "Для ЛЮБЫХ вычислений вы ОБЯЗАНЫ использовать инструменты add и multiply — никогда не считайте в уме. "
        "Получив задачу, немедленно вызовите нужный инструмент с правильными аргументами."
    )
)

web_agent = create_agent(
    model=model,
    tools=[web_search],
    name="web_expert",
    system_prompt=(
        "Вы - интернет-исследователь. "
        "Для ЛЮБОГО информационного запроса вы ОБЯЗАНЫ вызвать инструмент web_search с аргументом query. "
        "Никогда не отвечайте по памяти — всегда используйте инструмент."
    )
)

from langgraph_supervisor import create_supervisor

prompt = (
    "Вы - супервайзер команды. В вашем распоряжении два специалиста:\n"
    "- math_expert: решает математические задачи и вычисления. Инструмент: transfer_to_math_expert.\n"
    "- web_expert: ищет информацию в интернете (погода, новости, факты). Инструмент: transfer_to_web_expert.\n\n"
    "Строгие правила:\n"
    "1. Вы НИКОГДА не отвечаете на вопросы самостоятельно — всегда делегируйте нужному специалисту.\n"
    "2. Если запрос содержит несколько вопросов — разберите их по одному: сначала вызовите одного специалиста, "
    "дождитесь ответа, затем вызовите другого.\n"
    "3. Математика и вычисления → transfer_to_math_expert.\n"
    "4. Поиск в интернете, погода, новости → transfer_to_web_expert.\n"
    "5. Только после получения ответов от ВСЕХ нужных специалистов — сформируйте итоговый ответ пользователю."
)

math_search_workflow = create_supervisor(
    [web_agent, math_agent],
    model=model,
    output_mode="full_history",
    prompt=prompt,
)

app = math_search_workflow.compile()


result = app.invoke({
    "messages": [
        {
            "role": "user",
            "content": "Сколько будет 512*234? А Какая сегодня погода?"
        }
    ]
})

print(result)
with open("output.json", "w") as f:
    import json
    json.dump(result, f, indent=4, ensure_ascii=False, default=str)