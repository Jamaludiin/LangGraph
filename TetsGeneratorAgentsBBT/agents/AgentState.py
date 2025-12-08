from typing import TypedDict
# 2️⃣ Define State schema (REQUIRED for LangGraph v1)
# ----------------------------------------------------------
class AgentState(TypedDict):
    code_under_test: str
    test_plan: str
    test_case: str