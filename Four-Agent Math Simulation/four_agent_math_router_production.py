# filename: python3 four_agent_math_router_production.py

from typing import TypedDict, List
from langgraph.graph import StateGraph, END
import re

# ----------------------------------------------------------
# 1️⃣ Shared State
# ----------------------------------------------------------
class MathState(TypedDict):
    message: str
    operations: List[str]

# ----------------------------------------------------------
# 2️⃣ Deterministic Router (NO LLM)
# ----------------------------------------------------------
def router(state: MathState) -> MathState:
    text = state["message"].lower()
    operations = []

    if "add" in text or "+" in text:
        operations.append("addition")

    if "subtract" in text or "minus" in text or "-" in text:
        operations.append("subtraction")

    if "multiply" in text or "multiplied" in text or "*" in text:
        operations.append("multiplication")

    if "divide" in text or "divided" in text or "/" in text:
        operations.append("division")

    print("🧠 Router detected:", operations)

    return {
        "message": state["message"],
        "operations": operations
    }

# ----------------------------------------------------------
# 3️⃣ Helper: Extract numbers safely
# ----------------------------------------------------------
def extract_numbers(text: str):
    numbers = re.findall(r"-?\d+", text)
    return [int(n) for n in numbers]

# ----------------------------------------------------------
# 4️⃣ Addition Agent
# ----------------------------------------------------------
def addition_agent(state: MathState) -> MathState:
    nums = extract_numbers(state["message"])
    result = sum(nums)
    print(f"➕ Addition Agent: {nums} = {result}")
    return state

# ----------------------------------------------------------
# 5️⃣ Subtraction Agent
# ----------------------------------------------------------
def subtraction_agent(state: MathState) -> MathState:
    nums = extract_numbers(state["message"])
    if len(nums) >= 2:
        result = nums[0] - nums[1]
        print(f"➖ Subtraction Agent: {nums[0]} - {nums[1]} = {result}")
    return state

# ----------------------------------------------------------
# 6️⃣ Multiplication Agent
# ----------------------------------------------------------
def multiplication_agent(state: MathState) -> MathState:
    nums = extract_numbers(state["message"])
    result = 1
    for n in nums:
        result *= n
    print(f"✖ Multiplication Agent: {nums} = {result}")
    return state

# ----------------------------------------------------------
# 7️⃣ Division Agent
# ----------------------------------------------------------
def division_agent(state: MathState) -> MathState:
    nums = extract_numbers(state["message"])
    if len(nums) >= 2:
        if nums[1] == 0:
            print("➗ Division Agent: Error (division by zero)")
        else:
            result = nums[0] / nums[1]
            print(f"➗ Division Agent: {nums[0]} ÷ {nums[1]} = {result}")
    return state

# ----------------------------------------------------------
# 8️⃣ Conditional Routing Logic
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

# Router decides which operation to execute
graph.add_conditional_edges("router", route_operation)

# After each operation, go back to router for next one
graph.add_edge("addition", "router")
graph.add_edge("subtraction", "router")
graph.add_edge("multiplication", "router")
graph.add_edge("division", "router")

app = graph.compile()

# ----------------------------------------------------------
# 🔟 Run
# ----------------------------------------------------------
app.invoke({
    "message": "what is 45 divided by 9 and what is 45 multiplied by 9?"
})

print("✅ Done.")


"""
| Problem              | Cause                                     |
| -------------------- | ----------------------------------------- |
| Infinite loop        | Router recalculates operations every time |
| Recursion error      | operations list never empty               |
| Wrong multiplication | Extracting all numbers globally           |


"""
