"""
4️⃣ Mutational Testing Agent
This agent creates mutations of the original code (small changes) then asks the test generator to ensure the tests catch those mutations.

Example mutations:
    Change == to !=
    Remove boundary checks
    Replace numbers
    Invert booleans

Why?
    Guarantees strong test coverage.
"""

from agents.AgentState import AgentState
from agents.environmentVariables import llm

def mutational_testing_node(state: AgentState):
    response = llm.invoke(
        f"""Create mutations of the original code and ask the test generator to ensure the tests catch those mutations:
            \n\n{state['code_under_test']}
            \n\n{state['static_analyzer']}
            \n\n{state['test_plan']}
            \n\n{state['test_case']}
            \n\n{state['patch_suggestion']}
            \n\nCreate mutations of the original code and ask the test generator to ensure the tests catch those mutations:
            \n\nChange == to !=
            \n\nRemove boundary checks
            \n\nReplace numbers
            \n\nInvert booleans
            \n\nReturn only the pure python code, no other text or comments.
            \n\nno additional prefix or postfix or describtions
            \n\ndo not include the code these ``` or python keywords
            \n\nonly procide the code only no additional prefix or postfix or describtions
            \n\ndo not include the code these ``` or python keywords
            \n\nask the test generator to ensure the tests catch those mutations
        """
    )
    # Write the mutational testing to a file
    output_file_path = "/Users/LLM and HuggingFace/LangGraph/TetsGeneratorAgentsBBT/AgentFeedback/mutational_testing/mutational_testing.md"
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, "w") as file:
        file.write(response.content)
    return {"mutational_testing": response.content}