"""
6️⃣ Documentation Generator Agent
After test case generation, add a documentation generator agent that generates:
    test documentation
    test-report.md
    explanation of the logic
    how tests map to functions

Perfect for DevOps workflows.
"""

from agents.AgentState import AgentState
from agents.environmentVariables import llm

def documentation_generator_node(state: AgentState):
    response = llm.invoke(
        f"""Generate test documentation, test-report.md, explanation of the logic, and how tests map to functions:
            \n\n{state['code_under_test'], 
            state['test_plan'], state['test_case'], 
            state['static_analyzer'], 
            state['patch_suggestion'], state['execution'], 
            state['mutational_testing'], state['bug_predictor'], 
            state['test_plan_reviewer'], state['test_case_optimizer']}

        """
    )
    # Write the documentation generator to a file
    output_file_path = "/Users/LLM and HuggingFace/LangGraph/TetsGeneratorAgentsBBT/AgentFeedback/documentation_generator/documentation_generator.md"
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, "w") as file:
        file.write(response.content)
    return {"documentation_generator": response.content}