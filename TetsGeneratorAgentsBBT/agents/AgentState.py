from typing import TypedDict
# 2️⃣ Define State schema (REQUIRED for LangGraph v1)
# ----------------------------------------------------------
class AgentState(TypedDict):
    code_under_test: str
    test_plan: str
    test_case: str
    
    failure: str
    patch_suggestion: str
    execution: str
    documentation_generator: str
    coverage_enhancer: str
    mutational_testing: str
    bug_predictor: str
    test_plan_reviewer: str
    test_case_optimizer: str