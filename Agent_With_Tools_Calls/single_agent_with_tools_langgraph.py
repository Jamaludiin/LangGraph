# filename: python3 single_agent_with_tools_langgraph.py

from typing import TypedDict
import os
import re
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
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
    temperature=0,
    max_tokens=512
)

# ----------------------------------------------------------
# 2️⃣ Tools
# ----------------------------------------------------------

def extract_numbers(text: str):
    nums = re.findall(r"-?\d+\.?\d*", text)
    return [float(n) for n in nums]

@tool
def add(query: str) -> str:
    """Use for addition problems."""
    nums = extract_numbers(query)
    return str(sum(nums))

@tool
def subtract(query: str) -> str:
    """Use for subtraction problems."""
    nums = extract_numbers(query)
    if len(nums) < 2:
        return "Need at least two numbers."
    return str(nums[0] - nums[1])

@tool
def multiply(query: str) -> str:
    """Use for multiplication problems."""
    nums = extract_numbers(query)
    result = 1
    for n in nums:
        result *= n
    return str(result)

@tool
def divide(query: str) -> str:
    """Use for division problems."""
    nums = extract_numbers(query)
    if len(nums) < 2:
        return "Need two numbers."
    if nums[1] == 0:
        return "Division by zero error."
    return str(nums[0] / nums[1])

tools = [add, subtract, multiply, divide]

# Bind tools to LLM
llm_with_tools = llm.bind_tools(tools)

# ----------------------------------------------------------
# 3️⃣ Define minimal state
# ----------------------------------------------------------
class AgentState(TypedDict):
    input: str
    output: str

# ----------------------------------------------------------
# 4️⃣ Single Agent Node (Tool-aware)
# ----------------------------------------------------------
def agent_node(state: AgentState) -> AgentState:
    user_input = state["input"]

    response = llm_with_tools.invoke(
        [HumanMessage(content=user_input)]
    )

    # 👇 DEBUG: See raw model response
    print("\n🔍 Raw LLM response:", response)

    if response.tool_calls:
        print("\n🛠 Tool was called!")
        print("Tool Calls:", response.tool_calls)

        tool_call = response.tool_calls[0]
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]["query"]

        print("👉 Tool Name:", tool_name)
        print("👉 Tool Args:", tool_args)

        selected_tool = {t.name: t for t in tools}[tool_name]

        tool_result = selected_tool.invoke(tool_args)

        print("👉 Tool Result:", tool_result)

        final_response = llm_with_tools.invoke([
            HumanMessage(content=user_input),
            response,
            ToolMessage(
                content=tool_result,
                tool_call_id=tool_call["id"]
            )
        ])

        return {
            "input": user_input,
            "output": final_response.content
        }

    return {
        "input": user_input,
        "output": response.content
    }

# ----------------------------------------------------------
# 5️⃣ Build LangGraph
# ----------------------------------------------------------
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.set_entry_point("agent")
graph.add_edge("agent", END)

app = graph.compile()

# ----------------------------------------------------------
# 6️⃣ Run
# ----------------------------------------------------------
result = app.invoke(
    {"input": "What is 45 x 6?"}
    # Try:
    # {"input": "What is 45 divided by 9 and 45 multiplied by 9?"}
)

print("🤖 Agent:", result["output"])
print("✅ Done.")



"""
🎯 So The Execution Chain Is:
User input
    ↓
LLM decides tool name
    ↓
Your code reads response.tool_calls
    ↓
You manually call selected_tool.invoke(...)
    ↓
Python function executes
    ↓
Result returned
    ↓
You send result back to LLM
"""