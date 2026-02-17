🧭 4️⃣ Memory & State Design

Right now your state is:

class AgentState(TypedDict):
    input: str
    output: str


That’s too simple.

Next learn to store:

class AgentState(TypedDict):
    messages: list
    intermediate_steps: list
    iteration_count: int


LangGraph shines when managing complex state across nodes.