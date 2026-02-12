# filename: python3 single_agent_math_tools.py

import os
import re
from dotenv import load_dotenv
from typing import Annotated

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# ----------------------------------------------------------
# 1️⃣ Load environment
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
# 2️⃣ Helper: Extract numbers safely
# ----------------------------------------------------------
def extract_numbers(text: str):
    nums = re.findall(r"-?\d+\.?\d*", text)
    return [float(n) for n in nums]

# ----------------------------------------------------------
# 3️⃣ Tools
# ----------------------------------------------------------

@tool
def addition_tool(query: str) -> str:
    """Use this tool to solve addition problems."""
    nums = extract_numbers(query)
    result = sum(nums)
    return f"The sum of {nums} is {result}"

@tool
def subtraction_tool(query: str) -> str:
    """Use this tool to solve subtraction problems."""
    nums = extract_numbers(query)
    if len(nums) < 2:
        return "Need at least two numbers."
    result = nums[0] - nums[1]
    return f"{nums[0]} minus {nums[1]} is {result}"

@tool
def multiplication_tool(query: str) -> str:
    """Use this tool to solve multiplication problems."""
    nums = extract_numbers(query)
    result = 1
    for n in nums:
        result *= n
    return f"The product of {nums} is {result}"

@tool
def division_tool(query: str) -> str:
    """Use this tool to solve division problems."""
    nums = extract_numbers(query)
    if len(nums) < 2:
        return "Need at least two numbers."
    if nums[1] == 0:
        return "Error: Division by zero."
    result = nums[0] / nums[1]
    return f"{nums[0]} divided by {nums[1]} is {result}"

tools = [
    addition_tool,
    subtraction_tool,
    multiplication_tool,
    division_tool
]

# ----------------------------------------------------------
# 4️⃣ Create Simple ReAct Agent
# ----------------------------------------------------------
agent = create_react_agent(llm, tools)

# ----------------------------------------------------------
# 🔟 Run
# ----------------------------------------------------------
response = agent.invoke({
    "messages": [
        ("human", "What is 45 divided by 9?")
        # Try:
        # ("human", "What is 45 divided by 9 and what is 45 multiplied by 9?")
    ]
})

print(response["messages"][-1].content)
print("✅ Done.")
