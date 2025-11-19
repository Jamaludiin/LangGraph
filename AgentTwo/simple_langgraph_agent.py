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
class AgentState(TypedDict): # AgentState is like a notebook where your agent keeps track of things:
    input: str # what the user says
    chat_history: List[str] # what the agent has said or conversation memory (past chats)
    tool_output: str # what the tool has found or returned
    intermediate_steps: Annotated[list, operator.add] # list of steps the agent has taken (tools used, etc.)


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
# agent_node is like a brain that processes information and makes decisions:
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
