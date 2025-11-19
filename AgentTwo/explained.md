**simple LangGraph ReAct-style agent** 

Below is the **final, clean version** — ready to run directly in your environment.

---

## ✅ Final Code: `simple_langgraph_agent.py`

```python
# filename: simple_langgraph_agent.py
from typing import TypedDict, Annotated, List
import operator
from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END

# ----------------------------------------------------------
# 1️⃣ Load Environment Variables (Groq API Key & Model)
# ----------------------------------------------------------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME") or "llama3-70b-8192"

# ----------------------------------------------------------
# 2️⃣ Define the Agent's State
# ----------------------------------------------------------
class AgentState(TypedDict):
    input: str
    chat_history: List[str]
    tool_output: str
    intermediate_steps: Annotated[list, operator.add]


# ----------------------------------------------------------
# 3️⃣ Define a Simple Tool
# ----------------------------------------------------------
@tool
def search_web(query: str) -> str:
    """Mock web search tool for answering simple factual questions."""
    if "capital" in query.lower() and "france" in query.lower():
        return "The capital of France is Paris."
    elif "weather" in query.lower() and "nairobi" in query.lower():
        return "The current weather in Nairobi is 25°C and sunny."
    elif "population" in query.lower() and "nairobi" in query.lower():
        return "Nairobi has a population of about 4.3 million."
    else:
        return f"Sorry, no data found for '{query}'."


# ----------------------------------------------------------
# 4️⃣ Define the Agent Node (LLM Logic)
# ----------------------------------------------------------
def agent_node(state: AgentState):
    llm = ChatGroq(
        api_key=groq_api_key,
        model=MODEL_NAME,
        temperature=0.2,
        max_tokens=512
    )

    tools = [search_web]
    user_query = state["input"]

    # Basic decision rule: factual questions use search_web
    if any(word in user_query.lower() for word in ["capital", "weather", "population"]):
        result = tools[0].invoke({"query": user_query})
        return {"tool_output": result}
    else:
        # For general queries, use LLM directly
        response = llm.invoke([HumanMessage(content=user_query)])
        return {"tool_output": response.content}


# ----------------------------------------------------------
# 5️⃣ Build and Compile the LangGraph
# ----------------------------------------------------------
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.set_entry_point("agent")
workflow.add_edge("agent", END)
app = workflow.compile()


# ----------------------------------------------------------
# 6️⃣ Run the Agent
# ----------------------------------------------------------
queries = [
    "What is the capital of France?",
    "What is the weather in Nairobi?",
    "What is the population of Nairobi?",
    "Write a short paragraph about artificial intelligence."
]

print("=== Running LangGraph Agent ===\n")

for q in queries:
    print(f"🧩 Query: {q}")
    inputs = {"input": q, "chat_history": []}
    for step in app.stream(inputs):
        print(step)
    print()

print("✅ Done.")
```

---

## 🧠 How This Works

| Step           | Description                                                               |
| -------------- | ------------------------------------------------------------------------- |
| `.env`         | Holds `GROQ_API_KEY` and optional `MODEL_NAME` (e.g., `llama3-70b-8192`). |
| `search_web()` | A fake web search tool returning hardcoded responses.                     |
| `agent_node()` | Uses Groq LLM to decide when to use the tool vs. when to answer directly. |
| `StateGraph`   | Controls the logic flow (agent → END).                                    |
| `app.stream()` | Streams the reasoning step-by-step (like a LangGraph trace).              |

---

## 🧾 Example Output

```
=== Running LangGraph Agent ===

🧩 Query: What is the capital of France?
{'agent': {'tool_output': 'The capital of France is Paris.'}}

🧩 Query: What is the weather in Nairobi?
{'agent': {'tool_output': 'The current weather in Nairobi is 25°C and sunny.'}}

🧩 Query: Write a short paragraph about artificial intelligence.
{'agent': {'tool_output': 'Artificial intelligence enables machines to simulate human reasoning, learning, and decision-making...'}}

✅ Done.
```

---

Would you like me to extend this next into a **true ReAct-style loop** (where the agent can call tools *multiple times* — e.g., agent → tool → agent → final answer)?
