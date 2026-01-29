# filename: simple_langgraph_agent_with_memory.py
from typing import TypedDict, Annotated, List
import operator
from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END

# ----------------------------------------------------------
# 1️⃣ Load Environment Variables (Groq API Key & Model)
# ----------------------------------------------------------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME") or "llama3-70b-8192"

# ----------------------------------------------------------
# 2️⃣ Define the Agent's State
# ----------------------------------------------------------
class AgentState(TypedDict):
    input: str
    chat_history: List[str]
    tool_output: str
    intermediate_steps: Annotated[list, operator.add]


# ----------------------------------------------------------
# 3️⃣ Define a Simple Tool
# ----------------------------------------------------------
@tool
def search_web(query: str) -> str:
    """Mock web search tool for answering simple factual questions."""
    if "capital" in query.lower() and "france" in query.lower():
        return "The capital of France is Paris."
    elif "weather" in query.lower() and "nairobi" in query.lower():
        return "The current weather in Nairobi is 25°C and sunny."
    elif "population" in query.lower() and "nairobi" in query.lower():
        return "Nairobi has a population of about 4.3 million."
    else:
        return f"Sorry, no data found for '{query}'."


# ----------------------------------------------------------
# 4️⃣ Define the Agent Node (LLM Logic + Memory)
# ----------------------------------------------------------
def agent_node(state: AgentState):
    llm = ChatGroq(
        api_key=groq_api_key,
        model=MODEL_NAME,
        temperature=0.2,
        max_tokens=512
    )

    tools = [search_web]
    user_query = state["input"]
    chat_history = state.get("chat_history", [])

    # Simple decision rule for when to use tool
    if any(word in user_query.lower() for word in ["capital", "weather", "population"]):
        result = tools[0].invoke({"query": user_query})
        # append the result to the chat history
        chat_history.append(f"User: {user_query}")
        chat_history.append(f"Agent: {result}")
        # return the result and the chat history
        return {"tool_output": result, "chat_history": chat_history}
    else:
        # Use the chat history as context in LLM call
        context = "\n".join(chat_history[-4:])  # last few turns only
        full_prompt = f"Conversation so far:\n{context}\nUser: {user_query}\nAgent:"
        response = llm.invoke([HumanMessage(content=full_prompt)])
        chat_history.append(f"User: {user_query}")
        chat_history.append(f"Agent: {response.content}")
        return {"tool_output": response.content, "chat_history": chat_history}


# ----------------------------------------------------------
# 5️⃣ Build and Compile the LangGraph
# ----------------------------------------------------------
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.set_entry_point("agent")
workflow.add_edge("agent", END)
app = workflow.compile()


# ----------------------------------------------------------
# 6️⃣ Run the Agent
# ----------------------------------------------------------
queries = [
    "What is the capital of France?",
    "What is the weather in Nairobi?",
    "Can you remind me what city we talked about before?",
    "Write a short paragraph about artificial intelligence."
]

print("=== Running LangGraph Agent with Memory ===\n")

chat_history = []

for q in queries:
    print(f"🧩 Query: {q}")
    inputs = {"input": q, "chat_history": chat_history}
    for step in app.stream(inputs):
        chat_history = step["agent"]["chat_history"]
        print(step)
    print("\n--- Chat History ---")
    for line in chat_history[-6:]:  # Show last few exchanges
        print(line)
    print("\n")

print("✅ Done.")
