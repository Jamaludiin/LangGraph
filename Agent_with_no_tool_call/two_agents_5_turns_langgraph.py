# filename: python3 two_agents_5_turns_langgraph.py
# two agents introducing themselves for 5 turns
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
    max_tokens=150
)

# ----------------------------------------------------------
# 2️⃣ Define state
# ----------------------------------------------------------
class AgentState(TypedDict):
    message: str
    turn: int

# ----------------------------------------------------------
# 3️⃣ Agent One
# ----------------------------------------------------------
def agent_one(state: AgentState) -> AgentState:
    response = llm.invoke([
        HumanMessage(
            content="""
                        You are Agent One.
                        Reply shortly using this template ONLY:

                        Name:
                        Role:
                        Expertise:
                        Ask Agent Two a short question.
                        """
        )
    ])

    print(f"\n🧑‍💼 Agent One (Turn {state['turn']}):")
    print(response.content)

    return {
        "message": response.content,
        "turn": state["turn"] + 1
    }

# ----------------------------------------------------------
# 4️⃣ Agent Two
# ----------------------------------------------------------
def agent_two(state: AgentState) -> AgentState:
    response = llm.invoke([
        HumanMessage(
            content=f"""
                    You are Agent Two.
                    Reply shortly using this template ONLY:

                    Name:
                    Role:
                    Expertise:
                    Answer Agent One briefly.

                    Agent One said:
                    {state['message']}
                    """
        )
    ])

    print(f"\n🤖 Agent Two (Turn {state['turn']}):")
    print(response.content)

    return {
        "message": response.content,
        "turn": state["turn"]
    }

# ----------------------------------------------------------
# 5️⃣ Loop condition
# ----------------------------------------------------------
def should_continue(state: AgentState):
    if state["turn"] >= 5:
        return END
    return "agent_one"

# ----------------------------------------------------------
# 6️⃣ Build LangGraph
# ----------------------------------------------------------
graph = StateGraph(AgentState)

graph.add_node("agent_one", agent_one)
graph.add_node("agent_two", agent_two)

graph.set_entry_point("agent_one")
graph.add_edge("agent_one", "agent_two")
graph.add_conditional_edges("agent_two", should_continue)

app = graph.compile()

# ----------------------------------------------------------
# 7️⃣ Run
# ----------------------------------------------------------
app.invoke({
    "message": "",
    "turn": 1
})

print("\n✅ Done.")
