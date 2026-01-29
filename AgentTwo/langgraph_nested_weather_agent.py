# filename: langgraph_nested_weather_agent.py
# ok runing good now

from typing import TypedDict, Annotated, List
import operator
from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END

# ----------------------------------------------------------
# 1️⃣ Load Environment Variables
# ----------------------------------------------------------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME") or "llama3-70b-8192"

# ----------------------------------------------------------
# 2️⃣ Define Agent State
# ----------------------------------------------------------
class AgentState(TypedDict):
    input: str
    chat_history: List[str]
    tool_output: str
    intermediate_steps: Annotated[list, operator.add]

# ----------------------------------------------------------
# 3️⃣ TOOL 1: Search Web
# ----------------------------------------------------------
@tool
def search_web(query: str) -> str:
    """Mock web search tool."""
    if "weather" in query.lower() and "nairobi" in query.lower():
        return "Today's forecast shows 25°C and sunny skies in Nairobi."
    elif "population" in query.lower() and "nairobi" in query.lower():
        return "Nairobi has a population of around 4.3 million people."
    else:
        return f"No search results found for '{query}'."

# ----------------------------------------------------------
# 4️⃣ TOOL 2: Analyze Weather (depends on TOOL 1)
# ----------------------------------------------------------
@tool
def analyze_weather(city: str) -> str:
    """Analyze weather using the search_web tool."""
    search_result = search_web.invoke({"query": f"weather in {city}"})

    if "25" in search_result:
        return f"{city} is warm and sunny (25°C). A great day to go outside!"
    else:
        return f"Unable to determine the weather for {city}."

# ----------------------------------------------------------
# 5️⃣ Agent Node (Decision Logic + Memory)
# ----------------------------------------------------------
def agent_node(state: AgentState):
    llm = ChatGroq(
        api_key=groq_api_key,
        model=MODEL_NAME,
        temperature=0.2,
        max_tokens=512
    )

    user_query = state["input"]
    chat_history = state.get("chat_history", [])

    # --- Rule-based tool routing (simple & clear for teaching) ---
    if "analyze" in user_query.lower() and "weather" in user_query.lower():
        result = analyze_weather.invoke({"city": "Nairobi"})

    elif "weather" in user_query.lower():
        result = search_web.invoke({"query": user_query})

    else:
        context = "\n".join(chat_history[-4:])
        prompt = f"""
                    Conversation so far:
                    {context}

                    User: {user_query}
                    Agent:
                    """
        response = llm.invoke([HumanMessage(content=prompt)])
        result = response.content

    # Update memory
    chat_history.append(f"User: {user_query}")
    chat_history.append(f"Agent: {result}")

    return {
        "tool_output": result,
        "chat_history": chat_history
    }

# ----------------------------------------------------------
# 6️⃣ Build LangGraph Workflow
# ----------------------------------------------------------
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.set_entry_point("agent")
workflow.add_edge("agent", END)

app = workflow.compile()

# ----------------------------------------------------------
# 7️⃣ Run the Agent
# ----------------------------------------------------------
print("\n=== Running LangGraph Nested Agent ===\n")

chat_history = []

queries = [
    "Please analyze the weather in Nairobi.",
    "What is the population of Nairobi?",
    "Can you remind me what city we discussed?",
]

for q in queries:
    print(f"🧩 Query: {q}")
    inputs = {"input": q, "chat_history": chat_history}
    for step in app.stream(inputs):
        chat_history = step["agent"]["chat_history"]
        print(step["agent"]["tool_output"])
    print("\n--- Recent Memory ---")
    for line in chat_history[-6:]:
        print(line)
    print("\n")

print("✅ Done.")
