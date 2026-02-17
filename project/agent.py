# agent.py

import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, ToolMessage
from tools import ALL_TOOLS

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model=os.getenv("MODEL_NAME") or "llama3-70b-8192",
    temperature=0
)

llm_with_tools = llm.bind_tools(ALL_TOOLS)


def agent_node(state):
    messages = state["messages"]

    response = llm_with_tools.invoke(messages)

    messages.append(response)

    # If tool was called
    if response.tool_calls:
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            # Find correct tool
            tool = next(t for t in ALL_TOOLS if t.name == tool_name)

            # Execute tool
            tool_result = tool.invoke(tool_args)

            # Add tool response to messages
            messages.append(
                ToolMessage(
                    content=tool_result,
                    tool_call_id=tool_call["id"]
                )
            )

        # Re-invoke LLM with tool result
        final_response = llm_with_tools.invoke(messages)
        messages.append(final_response)

    return {"messages": messages}
