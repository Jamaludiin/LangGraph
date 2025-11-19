# python3 "/Users/LLM and HuggingFace/LangGraph/AgentTwo/lanGraphThreeAgents.py"
import os
from typing import TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchResults

# ----------------------------------------------------------
# 1️⃣ Load environment variables
# ----------------------------------------------------------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME") or "llama3-70b-8192"

# ----------------------------------------------------------
# 2️⃣ Define State schema (REQUIRED for LangGraph v1)
# ----------------------------------------------------------
class AgentState(TypedDict):
    topic: str
    research: str
    draft: str
    final: str

# ----------------------------------------------------------
# 3️⃣ LLM + Tools
# ----------------------------------------------------------
llm = ChatGroq(
    api_key=groq_api_key,
    model=MODEL_NAME,
    temperature=0.2,
)

search_tool = DuckDuckGoSearchResults(
    max_results=4,         # limit results
    results_separator="\n",
    verbose=True           # helpful for debugging
)

# ----------------------------------------------------------
# 4️⃣ Node Functions
# ----------------------------------------------------------
def researcher_node(state: AgentState):
    topic = state["topic"]
    result = "No results found."
    try:
        # DuckDuckGoSearchResults v0.2/v0.3 expects dict input
        result = search_tool.run({"query": topic, "max_results": 4})
    except Exception as e:
        print("DuckDuckGoSearchResults failed:", e)

    # Invoke the LLM with the research results 
    response = llm.invoke(
        f"Research the topic below and return detailed notes:\n\n"
        f"Topic: {topic}\n\nSearch Results: {result}"
    )

    return {"research": response.content}


def writer_node(state: AgentState):
    response = llm.invoke(
        f"Use the research notes below to write a clear, well-structured explanation:\n\n{state['research']}"
    )
    return {"draft": response.content}


def editor_node(state: AgentState):
    response = llm.invoke(
        f"Edit and improve the final draft. Fix grammar, clarity, and flow.\n\nDraft:\n{state['draft']}"
    )
    return {"final": response.content}

# ----------------------------------------------------------
# 5️⃣ Build LangGraph workflow
# ----------------------------------------------------------
workflow = StateGraph(AgentState)

workflow.add_node("researcher", researcher_node)
workflow.add_node("writer", writer_node)
workflow.add_node("editor", editor_node)

workflow.add_edge(START, "researcher")
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", "editor")
workflow.add_edge("editor", END)

app = workflow.compile()

# ----------------------------------------------------------
# 6️⃣ Run workflow
# ----------------------------------------------------------
if __name__ == "__main__":
    topic = "Impact of AI on software development"
    result = app.invoke({"topic": topic})
    print("\n=== FINAL OUTPUT ===\n")
    print(result["final"])
