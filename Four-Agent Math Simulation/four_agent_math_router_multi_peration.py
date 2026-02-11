# filename: python3 four_agent_math_router_multi_peration.py

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
    temperature=0, # temperature is a parameter that controls the randomness of the model's output, 0 is the most deterministic
    max_tokens=200
)

# ----------------------------------------------------------
# 2️⃣ Shared State
# ----------------------------------------------------------
class MathState(TypedDict):
    message: str
    operations: list[str]

# ----------------------------------------------------------
# 3️⃣ Router Agent (Decision Maker)
# ----------------------------------------------------------
def router(state: MathState) -> MathState:
    user_input = state["message"]

    response = llm.invoke([
        HumanMessage(content=f"""
From the user input below, detect ALL required operations.

Possible operations:
- addition
- subtraction
- multiplication
- division

Respond with a comma-separated list.
Example:
addition,multiplication

User input:
{user_input}
""")
    ])

    ops = response.content.strip().lower()
    operations = [op.strip() for op in ops.split(",")]

    print("🧠 Router decided:", operations)

    return {
        "message": user_input,
        "operations": operations
    }


# ----------------------------------------------------------
# 4️⃣ Addition Agent
# ----------------------------------------------------------
def addition_agent(state: MathState) -> MathState:
    response = llm.invoke([
        HumanMessage(content=f"""
You are an Addition Agent.
Solve the following addition problem clearly:

{state['message']}
""")
    ])

    print("➕ Addition Agent:", response.content)
    return state


# ----------------------------------------------------------
# 5️⃣ Subtraction Agent
# ----------------------------------------------------------
def subtraction_agent(state: MathState) -> MathState:
    response = llm.invoke([
        HumanMessage(content=f"""
You are a Subtraction Agent.
Solve the following subtraction problem clearly:

{state['message']}
""")
    ])

    print("➖ Subtraction Agent:", response.content)
    return state


# ----------------------------------------------------------
# 6️⃣ Multiplication Agent
# ----------------------------------------------------------
def multiplication_agent(state: MathState) -> MathState:
    response = llm.invoke([
        HumanMessage(content=f"""
You are a Multiplication Agent.
Solve the following multiplication problem clearly:

{state['message']}
""")
    ])

    print("✖ Multiplication Agent:", response.content)
    return state


# ----------------------------------------------------------
# 7️⃣ Division Agent
# ----------------------------------------------------------
def division_agent(state: MathState) -> MathState:
    response = llm.invoke([
        HumanMessage(content=f"""
You are a Division Agent.
Solve the following division problem clearly:

{state['message']}
""")
    ])

    print("➗ Division Agent:", response.content)
    return state


# ----------------------------------------------------------
# 8️⃣ Conditional Edge Logic
# ----------------------------------------------------------
def route_operation(state: MathState):
    if not state["operations"]:
        return END

    next_op = state["operations"].pop(0)

    return next_op


# ----------------------------------------------------------
# 9️⃣ Build Graph
# ----------------------------------------------------------
graph = StateGraph(MathState)

graph.add_node("router", router)
graph.add_node("addition", addition_agent)
graph.add_node("subtraction", subtraction_agent)
graph.add_node("multiplication", multiplication_agent)
graph.add_node("division", division_agent)

graph.set_entry_point("router")

# Conditional branching happens here 👇
graph.add_conditional_edges("router", route_operation)

graph.add_edge("addition", END)
graph.add_edge("subtraction", END)
graph.add_edge("multiplication", END)
graph.add_edge("division", END)

app = graph.compile()

# ----------------------------------------------------------
# 🔟 Run
# ----------------------------------------------------------
app.invoke({
    #"message": "What is 45 divided by 9?"
    "message": "what is 45 divided by 9? and what is 45 multiplied by 9?"
})
# "what is 45 divided by 9? and "what is 45 multiplied by 9?" should both work

print("✅ Done.")



"""
🎓 Deep Architectural Insight

This is what you just discovered:

LLMs are BAD for control flow.
LLMs are GOOD for language generation.

Use:

Python for logic

LLM for reasoning / text

That separation makes professional agent systems stable.
"""


"""
🚨 Another BIG Issue in Your Graph

Right now you have:

graph.add_edge("addition", END)
graph.add_edge("subtraction", END)
graph.add_edge("multiplication", END)
graph.add_edge("division", END)


So even if routing worked,
each agent goes directly to END.

You need:

graph.add_edge("addition", "router")
graph.add_edge("subtraction", "router")
graph.add_edge("multiplication", "router")
graph.add_edge("division", "router")


Otherwise multi-operation will NEVER loop.

"""
