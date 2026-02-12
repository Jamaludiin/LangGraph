# filename: python3 one_agent_four_tools_math.py
# error GRAPH_RECURSION_LIMIT reached
# no langgraph only langchain with tools calls (create_react_agent)
import os
import re
from dotenv import load_dotenv
from typing import Annotated

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

# ----------------------------------------------------------
# 1️⃣ Load Environment
# ----------------------------------------------------------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME") or "llama3-70b-8192"

llm = ChatGroq(
    api_key=groq_api_key,
    model=MODEL_NAME,
    temperature=0
)

# ----------------------------------------------------------
# 2️⃣ Helper: Extract Numbers
# ----------------------------------------------------------
def extract_numbers(text: str):
    nums = re.findall(r"-?\d+", text)
    return [int(n) for n in nums]

# ----------------------------------------------------------
# 3️⃣ Define Four Tools
# ----------------------------------------------------------

@tool
def add(problem: str) -> str:
    """Use this tool for addition problems."""
    nums = extract_numbers(problem)
    result = sum(nums)
    return f"The addition result is: {result}"

@tool
def subtract(problem: str) -> str:
    """Use this tool for subtraction problems."""
    nums = extract_numbers(problem)
    if len(nums) >= 2:
        result = nums[0] - nums[1]
        return f"The subtraction result is: {result}"
    return "Not enough numbers for subtraction."

@tool
def multiply(problem: str) -> str:
    """Use this tool for multiplication problems."""
    nums = extract_numbers(problem)
    result = 1
    for n in nums:
        result *= n
    return f"The multiplication result is: {result}"

@tool
def divide(problem: str) -> str:
    """Use this tool for division problems."""
    nums = extract_numbers(problem)
    if len(nums) >= 2:
        if nums[1] == 0:
            return "Error: Cannot divide by zero."
        result = nums[0] / nums[1]
        return f"The division result is: {result}"
    return "Not enough numbers for division."

# ----------------------------------------------------------
# 4️⃣ Create Agent with Tools
# ----------------------------------------------------------

tools = [add, subtract, multiply, divide]

agent = create_react_agent(llm, tools)

# ----------------------------------------------------------
# 5️⃣ Run
# ----------------------------------------------------------

response = agent.invoke({
    "messages": [
        HumanMessage(content="What is 45 divided by 9?")
    ]
})

print("🤖 Agent Response:")
print(response["messages"][-1].content)

print("✅ Done.")
