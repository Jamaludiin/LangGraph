"""
1️⃣ Code Reader Agent
Read the code and extract the necessary information to generate a test plan.
"""

from agents.AgentState import AgentState
from agents.environmentVariables import llm


def code_reader_node(state: AgentState):
        # Read the code from the put.py file
    with open("/Users/LLM and HuggingFace/LangGraph/TetsGeneratorAgentsBBT/programUnderTest/put.py", "r") as file:
        code_under_test = file.read()
        
    return {"code_under_test": code_under_test}

