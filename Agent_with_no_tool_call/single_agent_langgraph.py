# filename: python3 single_agent_langgraph.py
# simple single agent – input -> output (LangGraph correct)
# working correctly

from typing import TypedDict
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

# ----------------------------------------------------------
# 1️⃣ Load environment
# ----------------------------------------------------------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME") or "llama3-70b-8192"

llm = ChatGroq(
    api_key=groq_api_key,
    model=MODEL_NAME,
    temperature=0.2,
    max_tokens=512
)

# ----------------------------------------------------------
# 2️⃣ Define minimal state
# ----------------------------------------------------------
class AgentState(TypedDict):
    input: str
    output: str

# ----------------------------------------------------------
# 3️⃣ Agent node (THIS is the key fix)
# ----------------------------------------------------------
def agent_node(state: AgentState) -> AgentState:
    user_input = state["input"]

    response = llm.invoke(
        [HumanMessage(content=user_input)]
    )

    return {
        "input": user_input,
        "output": response.content
    }

# ----------------------------------------------------------
# 4️⃣ Build LangGraph
# ----------------------------------------------------------
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.set_entry_point("agent")
graph.add_edge("agent", END)

app = graph.compile()

# ----------------------------------------------------------
# 5️⃣ Run
# ----------------------------------------------------------
result = app.invoke(
    {"input": "What is the capital of France?"}
)

print("🤖 Agent:", result["output"])
print("✅ Done.")
