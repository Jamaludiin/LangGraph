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
            \n\n{state['test_case']}
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
    return {"mutational_testing": response.content}