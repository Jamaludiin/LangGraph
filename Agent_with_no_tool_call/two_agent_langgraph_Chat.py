# filename: python3 two_agent_langgraph_Chat.py
# simulate two agents asking & answering each other
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
# 2️⃣ Define shared state
# ----------------------------------------------------------
class AgentState(TypedDict):
    message: str

# ----------------------------------------------------------
# 3️⃣ Agent 1: Asks a question
# ----------------------------------------------------------
def agent_one(state: AgentState) -> AgentState:
    question = state["message"]

    response = llm.invoke([
        HumanMessage(content=f"You are Agent One. Ask a clear question:\n{question}")
    ])

    print("🧑‍💼 Agent One:", response.content)

    return {"message": response.content}

# ----------------------------------------------------------
# 4️⃣ Agent 2: Answers Agent 1
# ----------------------------------------------------------
def agent_two(state: AgentState) -> AgentState:
    question_from_agent_one = state["message"]

    response = llm.invoke([
        HumanMessage(
            content=f"You are Agent Two. Answer the following question clearly:\n{question_from_agent_one}"
        )
    ])

    print("🤖 Agent Two:", response.content)

    return {"message": response.content}

# ----------------------------------------------------------
# 5️⃣ Build LangGraph
# ----------------------------------------------------------
graph = StateGraph(AgentState)
graph.add_node("agent_one", agent_one)
graph.add_node("agent_two", agent_two)

graph.set_entry_point("agent_one")
graph.add_edge("agent_one", "agent_two")
graph.add_edge("agent_two", END)

app = graph.compile()

# ----------------------------------------------------------
# 6️⃣ Run
# ----------------------------------------------------------
app.invoke({
    "message": "Introduce yourself and ask a question about artificial intelligence."
})

print("✅ Done.")



"""
    Initial message
        ↓
    Agent One (asks question)
        ↓
    Agent Two (answers Agent One)
        ↓
    END
"""
