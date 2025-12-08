# python3 "/Users/LLM and HuggingFace/LangGraph/TetsGeneratorAgents/testCasesGenerator.py"
# source "/Users/LLM and HuggingFace/.venv/bin/activate"




from typing import TypedDict
from langgraph.graph import StateGraph, START, END

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

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
    test_plan: str
    test_case: str

# ----------------------------------------------------------
# 3️⃣ LLM + Tools
# ----------------------------------------------------------
llm = ChatGroq(
    api_key=groq_api_key,
    model=MODEL_NAME,
    temperature=0.2,
)


# ----------------------------------------------------------
# 4️⃣ Node Functions
# ----------------------------------------------------------

# read the code from the put.py file
with open("/Users/LLM and HuggingFace/LangGraph/TetsGeneratorAgents/put.py", "r") as file:
    code_under_test = file.read()


def test_plan_node(state: AgentState):
    response = llm.invoke(
        f"Generate a test plan (positive and negative test cases) for the following code:\n\n{code_under_test}"
    )
    return {"test_plan": response.content}

# accepts the test plan and generates the test case
def test_case_node(state: AgentState):
    response = llm.invoke(
        f"Generate only a pure python test case using pytest style for the following code and plan:\n\n{code_under_test}\n\n{state['test_plan']}"
    )
    return {"test_case": response.content}
# ----------------------------------------------------------
# 5️⃣ Build LangGraph workflow
# ----------------------------------------------------------

workflow = StateGraph(AgentState)
workflow.add_node("test_plan_node", test_plan_node)
workflow.add_node("test_case_node", test_case_node)
workflow.add_edge(START, "test_plan_node")
workflow.add_edge("test_plan_node", "test_case_node")
workflow.add_edge("test_case_node", END)
app = workflow.compile()

# ----------------------------------------------------------
# 6️⃣ Run workflow
# ----------------------------------------------------------
if __name__ == "__main__":
    
    result = app.invoke({})
    print("\n=== FINAL OUTPUT ===\n")
    print(result["test_case"])


    # write the test case to a file
    with open("/Users/LLM and HuggingFace/LangGraph/TetsGeneratorAgents/test_case.py", "w") as file:
        file.write(result["test_case"])