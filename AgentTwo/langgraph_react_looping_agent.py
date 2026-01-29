from typing import TypedDict, Annotated, List
import operator
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END

# ----------------------------------------------------------
# 1️⃣ Load Environment Variables
# ----------------------------------------------------------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME") or "llama3-70b-8192"

llm = ChatGroq(
    api_key=groq_api_key,
    model=MODEL_NAME,
    temperature=0.2,
    max_tokens=512
)

# ----------------------------------------------------------
# 2️⃣ Agent State (Supports ReAct + Looping)
# ----------------------------------------------------------
class AgentState(TypedDict):
    input: str
    messages: Annotated[List, operator.add]
    next_step: str

# ----------------------------------------------------------
# 3️⃣ Tools
# ----------------------------------------------------------
@tool
def search_web(query: str) -> str:
    """Mock web search tool."""
    if "weather" in query.lower() and "nairobi" in query.lower():
        return "Nairobi weather: 25°C and sunny."
    elif "population" in query.lower() and "nairobi" in query.lower():
        return "Nairobi population is about 4.3 million."
    else:
        return "No relevant search results found."

@tool
def analyze_weather(city: str) -> str:
    """Analyze weather using search results."""
    result = search_web.invoke({"query": f"weather in {city}"})
    if "25" in result:
        return f"{city} is warm and sunny (25°C)."
    return f"Weather data for {city} is unclear."

TOOLS = {
    "search_web": search_web,
    "analyze_weather": analyze_weather
}

# ----------------------------------------------------------
# 4️⃣ ReAct Reasoning Node (LLM decides)
# ----------------------------------------------------------
def react_reasoner(state: AgentState):
    messages = state["messages"]

    prompt = f"""
                You are a ReAct agent.

                You must respond in ONE of the following formats ONLY:

                Thought: <your reasoning>
                Action: <tool_name>[<input>]

                OR

                Thought: <your reasoning>
                Final: <final answer>

                Available tools:
                - search_web
                - analyze_weather
                """

    # this is the prompt that the llm will use to reason about the user's question
    response = llm.invoke(messages + [HumanMessage(content=prompt)])
    messages.append(response)# same like chat history
    # if the response contains "Action:" then the next step is to execute the tool
    # otherwise the next step is to end the conversation
    if "Action:" in response.content:
        return {"messages": messages, "next_step": "tool"}
    else:
        # if the response contains "Final:" then the next step is to end the conversation
        return {"messages": messages, "next_step": "end"}

# ----------------------------------------------------------
# 5️⃣ Tool Executor Node
# ----------------------------------------------------------
def tool_executor(state: AgentState):
    last_message = state["messages"][-1].content # this is the last message in the chat history

    action_line = [l for l in last_message.splitlines() if l.startswith("Action:")][0] # this is the line that contains the tool name and the input
    tool_name, tool_input = action_line.replace("Action:", "").strip().split("[", 1) # this is the tool name and the input
    tool_input = tool_input.rstrip("]")

    tool = TOOLS.get(tool_name) # this is the tool that will be used to execute the input
    observation = tool.invoke({"query": tool_input}) if tool_name == "search_web" else tool.invoke({"city": tool_input}) # this is the observation that will be returned by the tool

    state["messages"].append(
        AIMessage(content=f"Observation: {observation}")
    )

    return {"messages": state["messages"], "next_step": "reason"}

# ----------------------------------------------------------
# 6️⃣ Build Looping LangGraph
# ----------------------------------------------------------
workflow = StateGraph(AgentState)

workflow.add_node("reason", react_reasoner)
workflow.add_node("tool", tool_executor)

workflow.set_entry_point("reason")

workflow.add_edge("tool", "reason")

workflow.add_conditional_edges(
    "reason",
    lambda state: state["next_step"],
    {
        "tool": "tool",
        "end": END
    }
)

app = workflow.compile()

# ----------------------------------------------------------
# 7️⃣ Run the Agent
# ----------------------------------------------------------
print("\n=== ReAct LangGraph Agent (Looping) ===\n")

inputs = {
    "input": "Analyze the weather in Nairobi",
    "messages": [HumanMessage(content="Analyze the weather in Nairobi")],
}

for step in app.stream(inputs):
    for k, v in step.items():
        if k == "reason" or k == "tool":
            print(v["messages"][-1].content)

print("\n✅ Done.")
