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
        "bla bla bla"
    )


from langchain.agents import create_agent

math_agent = create_agent(
    model=model,
    tools=[add, multiply],
    name="math_expert",
    system_prompt="Вы - эксперт по математике. У вас есть доступ к двум инструментам: add и multiply. Вызывайте эти инструменты для выполнения арифметических операций на основе вопроса пользователя."
)

research_agent = create_agent(
    model=model,
    tools=[web_search],
    name="research_expert",
    system_prompt="Вы - исследователь с доступом к одному инструменту: web_search. Вызывайте этот инструмент для поиска информации по вопросу пользователя."
)

from langgraph_supervisor import create_supervisor

prompt = (
    "Вы - супервайзер команды, управляющий экспертом по исследованиям и экспертом по математике. "
    "У вас есть доступ к двум инструментам: transfer_to_research_expert и transfer_to_math_expert. "
    "Для текущих событий вы можете вызвать инструмент transfer_to_research_expert. "
    "Для математических задач вы можете вызвать инструмент transfer_to_math_expert."
)

math_search_workflow = create_supervisor(
    [research_agent, math_agent],
    model=model,
    output_mode="full_history",
    prompt=prompt,
)

app = math_search_workflow.compile()


result = app.invoke({
    "messages": [
        {
            "role": "user",
            "content": "Сколько будет 2 + 2 и что такое квантовая запутанность?"
        }
    ]
})

print(result)
with open("output.json", "w") as f:
    import json
    json.dump(result, f, indent=4, ensure_ascii=False, default=str)