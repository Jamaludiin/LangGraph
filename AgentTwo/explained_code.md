# 🧠 LangGraph Agent with Memory - Complete Beginner's Guide

This guide explains every line of the `simple_langgraph_agent_with_memory.py` code. By the end, you'll understand how to build an AI agent that can remember conversations!

---

## 📋 Table of Contents

1. [What This Code Does](#what-this-code-does)
2. [The Imports](#1️⃣-the-imports-lines-1-10)
3. [Loading Environment Variables](#2️⃣-loading-environment-variables-lines-12-17)
4. [Defining Agent State](#3️⃣-defining-agent-state-lines-19-26)
5. [Creating a Tool](#4️⃣-creating-a-tool-lines-29-42)
6. [The Agent Logic](#5️⃣-the-agent-logic-lines-45-75)
7. [Building the Graph](#6️⃣-building-the-graph-lines-78-85)
8. [Running the Agent](#7️⃣-running-the-agent-lines-88-113)
9. [Sample Output Explained](#sample-output-explained)

---

## What This Code Does

This code creates an **AI chatbot** that can:
- ✅ Answer questions using a **mock tool** (like a fake web search)
- ✅ **Remember** previous conversations
- ✅ Use an **LLM** (Large Language Model) for general questions
- ✅ Decide **when to use tools** vs. when to use the LLM

---

## 1️⃣ The Imports (Lines 1-10)

```python
# filename: simple_langgraph_agent_with_memory.py
from typing import TypedDict, Annotated, List
import operator
from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
```

### Line-by-Line Explanation:

| Line | Code | What It Does |
|------|------|--------------|
| 1 | `# filename: ...` | Just a comment naming the file |
| 2 | `from typing import TypedDict, Annotated, List` | Imports type hints for defining data structures |
| 3 | `import operator` | Imports operator module (used for combining lists) |
| 4 | `from dotenv import load_dotenv` | Loads environment variables from a `.env` file |
| 5 | `import os` | Access environment variables |
| 7 | `from langchain_groq import ChatGroq` | The LLM client to talk to Groq's API |
| 8 | `from langchain_core.messages import HumanMessage` | Format messages for the LLM |
| 9 | `from langchain_core.tools import tool` | Decorator to create tools |
| 10 | `from langgraph.graph import StateGraph, END` | Core LangGraph components |

### 💡 Key Concepts:
- **TypedDict**: A dictionary with specific keys and types (like a schema)
- **StateGraph**: The main LangGraph class for building agent workflows
- **END**: A special node that marks the end of the graph

---

## 2️⃣ Loading Environment Variables (Lines 12-17)

```python
# ----------------------------------------------------------
# 1️⃣ Load Environment Variables (Groq API Key & Model)
# ----------------------------------------------------------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME") or "llama3-70b-8192"
```

### Line-by-Line Explanation:

| Line | Code | What It Does |
|------|------|--------------|
| 15 | `load_dotenv()` | Reads the `.env` file and loads variables into the environment |
| 16 | `groq_api_key = os.getenv("GROQ_API_KEY")` | Gets your Groq API key from environment |
| 17 | `MODEL_NAME = os.getenv("MODEL_NAME") or "llama3-70b-8192"` | Gets model name, defaults to LLaMA 3 70B |

### 💡 Why This Matters:
- **Never hardcode API keys** in your code!
- `.env` files keep secrets safe and out of version control
- The `or` provides a fallback if the variable isn't set

### 📄 Your `.env` file should look like:
```
GROQ_API_KEY=gsk_your_api_key_here
MODEL_NAME=llama-3.1-8b-instant
```

---

## 3️⃣ Defining Agent State (Lines 19-26)

```python
# ----------------------------------------------------------
# 2️⃣ Define the Agent's State
# ----------------------------------------------------------
class AgentState(TypedDict):
    input: str
    chat_history: List[str]
    tool_output: str
    intermediate_steps: Annotated[list, operator.add]
```

### Line-by-Line Explanation:

| Line | Code | What It Does |
|------|------|--------------|
| 22 | `class AgentState(TypedDict):` | Defines a typed dictionary for agent state |
| 23 | `input: str` | The user's current question |
| 24 | `chat_history: List[str]` | List of previous messages (the "memory") |
| 25 | `tool_output: str` | The response from tools or LLM |
| 26 | `intermediate_steps: Annotated[list, operator.add]` | Steps taken during processing (auto-combines lists) |

### 💡 Understanding State:

Think of **state** as a **backpack** the agent carries around:

```
┌─────────────────────────────────────┐
│           AgentState                │
├─────────────────────────────────────┤
│ input: "What is the capital?"       │  ← Current question
│ chat_history: ["User: Hi", ...]     │  ← Memory of past talks
│ tool_output: "Paris is the capital" │  ← Agent's answer
│ intermediate_steps: [...]           │  ← Processing steps
└─────────────────────────────────────┘
```

### 💡 What is `Annotated[list, operator.add]`?

This tells LangGraph: "When you update `intermediate_steps`, **add** to the existing list instead of replacing it."

---

## 4️⃣ Creating a Tool (Lines 29-42)

```python
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
```

### Line-by-Line Explanation:

| Line | Code | What It Does |
|------|------|--------------|
| 32 | `@tool` | Decorator that turns a function into a LangChain tool |
| 33 | `def search_web(query: str) -> str:` | Function that takes a query string and returns a string |
| 34 | `"""Mock web search..."""` | Docstring (describes what the tool does) |
| 35-42 | `if/elif/else` | Simple pattern matching to return fake "search results" |

### 💡 How Tools Work:

```
┌──────────────┐     ┌────────────────┐     ┌──────────────┐
│  User asks:  │ ──▶ │  Tool checks:  │ ──▶ │   Returns:   │
│ "capital of  │     │ "capital" AND  │     │  "Paris..."  │
│   France?"   │     │  "france" in   │     │              │
│              │     │    query?      │     │              │
└──────────────┘     └────────────────┘     └──────────────┘
```

### 💡 Why Mock Tools?

In real applications, you'd connect to:
- Real web search APIs
- Databases
- Weather services
- Calculators

This mock tool demonstrates the **pattern** without needing external services.

---

## 5️⃣ The Agent Logic (Lines 45-75)

This is the **brain** of the agent! Let's break it down:

### Part A: Setup (Lines 48-58)

```python
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
```

| Line | Code | What It Does |
|------|------|--------------|
| 48 | `def agent_node(state: AgentState):` | Defines the agent function that receives state |
| 49-54 | `llm = ChatGroq(...)` | Creates the LLM client with settings |
| 52 | `temperature=0.2` | Low = more focused answers (high = more creative) |
| 53 | `max_tokens=512` | Maximum length of response |
| 56 | `tools = [search_web]` | List of available tools |
| 57 | `user_query = state["input"]` | Gets the user's question from state |
| 58 | `chat_history = state.get("chat_history", [])` | Gets history, or empty list if none |

### Part B: Decision Logic (Lines 60-67)

```python
    # Simple decision rule for when to use tool
    if any(word in user_query.lower() for word in ["capital", "weather", "population"]):
        result = tools[0].invoke({"query": user_query})
        # append the result to the chat history
        chat_history.append(f"User: {user_query}")
        chat_history.append(f"Agent: {result}")
        # return the result and the chat history
        return {"tool_output": result, "chat_history": chat_history}
```

| Line | Code | What It Does |
|------|------|--------------|
| 61 | `if any(word in user_query.lower() ...)` | Checks if query contains trigger words |
| 62 | `result = tools[0].invoke(...)` | Calls the search_web tool |
| 64-65 | `chat_history.append(...)` | Saves the exchange to memory |
| 67 | `return {...}` | Returns updated state |

### 💡 Decision Flow:

```
                    ┌─────────────────────┐
                    │   User Question     │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │ Contains "capital", │
                    │ "weather", or       │
                    │ "population"?       │
                    └──────────┬──────────┘
                               │
              ┌────────────────┴────────────────┐
              │ YES                         NO  │
              ▼                                 ▼
    ┌─────────────────┐              ┌─────────────────┐
    │   Use TOOL      │              │    Use LLM      │
    │  (search_web)   │              │  (ChatGroq)     │
    └─────────────────┘              └─────────────────┘
```

### Part C: LLM Fallback (Lines 68-75)

```python
    else:
        # Use the chat history as context in LLM call
        context = "\n".join(chat_history[-4:])  # last few turns only
        full_prompt = f"Conversation so far:\n{context}\nUser: {user_query}\nAgent:"
        response = llm.invoke([HumanMessage(content=full_prompt)])
        chat_history.append(f"User: {user_query}")
        chat_history.append(f"Agent: {response.content}")
        return {"tool_output": response.content, "chat_history": chat_history}
```

| Line | Code | What It Does |
|------|------|--------------|
| 70 | `context = "\n".join(chat_history[-4:])` | Gets last 4 messages as context |
| 71 | `full_prompt = f"..."` | Builds prompt with history + question |
| 72 | `response = llm.invoke(...)` | Calls the LLM |
| 73-74 | `chat_history.append(...)` | Saves to memory |
| 75 | `return {...}` | Returns updated state |

### 💡 Memory in Action:

When asked "Can you remind me what city we talked about before?":

```
full_prompt = """
Conversation so far:
User: What is the capital of France?
Agent: The capital of France is Paris.
User: What is the weather in Nairobi?
Agent: The current weather in Nairobi is 25°C and sunny.
User: Can you remind me what city we talked about before?
Agent:
"""
```

The LLM sees the history and responds: **"We previously discussed Paris, the capital of France."**

---

## 6️⃣ Building the Graph (Lines 78-85)

```python
# ----------------------------------------------------------
# 5️⃣ Build and Compile the LangGraph
# ----------------------------------------------------------
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.set_entry_point("agent")
workflow.add_edge("agent", END)
app = workflow.compile()
```

### Line-by-Line Explanation:

| Line | Code | What It Does |
|------|------|--------------|
| 81 | `workflow = StateGraph(AgentState)` | Creates a new graph with our state schema |
| 82 | `workflow.add_node("agent", agent_node)` | Adds our agent function as a node |
| 83 | `workflow.set_entry_point("agent")` | Sets "agent" as the starting point |
| 84 | `workflow.add_edge("agent", END)` | Connects agent node to END (finish) |
| 85 | `app = workflow.compile()` | Compiles into a runnable application |

### 💡 Visual Graph:

```
    ┌─────────┐
    │  START  │
    └────┬────┘
         │
         ▼
    ┌─────────┐
    │  agent  │  ← Our agent_node function
    └────┬────┘
         │
         ▼
    ┌─────────┐
    │   END   │
    └─────────┘
```

This is a **simple linear graph**. More complex agents have multiple nodes with conditional edges!

---

## 7️⃣ Running the Agent (Lines 88-113)

```python
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
```

### Line-by-Line Explanation:

| Line | Code | What It Does |
|------|------|--------------|
| 91-96 | `queries = [...]` | List of test questions |
| 100 | `chat_history = []` | Initialize empty memory |
| 102 | `for q in queries:` | Loop through each question |
| 104 | `inputs = {...}` | Prepare input state |
| 105 | `for step in app.stream(inputs):` | Run the graph, streaming results |
| 106 | `chat_history = step["agent"]["chat_history"]` | Update memory from response |
| 109 | `for line in chat_history[-6:]:` | Print last 6 messages |

### 💡 The Magic of Memory:

```python
# Question 1: chat_history = []
# After Q1:   chat_history = ["User: What is the capital...", "Agent: Paris..."]

# Question 2: uses previous chat_history
# After Q2:   chat_history = [...Q1..., "User: What is the weather...", "Agent: 25°C..."]

# Question 3: "remind me what city?" → sees Paris in history!
```

---

## Sample Output Explained

Here's what the agent produced:

### Query 1: "What is the capital of France?"
```
🧩 Query: What is the capital of France?
{'agent': {'tool_output': 'The capital of France is Paris.', 
           'chat_history': ['User: What is the capital of France?', 
                           'Agent: The capital of France is Paris.']}}
```
**Decision**: Contains "capital" → Uses **TOOL** → Returns mock data

---

### Query 2: "What is the weather in Nairobi?"
```
🧩 Query: What is the weather in Nairobi?
{'agent': {'tool_output': 'The current weather in Nairobi is 25°C and sunny.', 
           'chat_history': [...previous..., 
                           'User: What is the weather in Nairobi?', 
                           'Agent: The current weather in Nairobi is 25°C and sunny.']}}
```
**Decision**: Contains "weather" → Uses **TOOL** → Returns mock weather

---

### Query 3: "Can you remind me what city we talked about before?"
```
🧩 Query: Can you remind me what city we talked about before?
{'agent': {'tool_output': 'We previously discussed Paris, the capital of France.', 
           'chat_history': [...previous..., 
                           'User: Can you remind me what city we talked about before?', 
                           'Agent: We previously discussed Paris, the capital of France.']}}
```
**Decision**: No trigger words → Uses **LLM** → LLM sees history and remembers Paris! 🎉

---

### Query 4: "Write a short paragraph about artificial intelligence."
```
🧩 Query: Write a short paragraph about artificial intelligence.
{'agent': {'tool_output': 'Artificial intelligence (AI) refers to the development 
           of computer systems that can perform tasks that typically require 
           human intelligence...', 
           'chat_history': [...]}}
```
**Decision**: No trigger words → Uses **LLM** → Generates creative response

---

## 🎓 Key Takeaways

| Concept | What You Learned |
|---------|------------------|
| **State** | A TypedDict that carries data between nodes |
| **Tools** | Functions the agent can call for specific tasks |
| **Memory** | Passing `chat_history` enables conversation recall |
| **Routing** | Simple if/else decides tool vs. LLM |
| **Graph** | Nodes connected by edges define the workflow |

---

## 🚀 Next Steps

1. **Add more tools** (calculator, database lookup, real APIs)
2. **Add conditional edges** (route to different nodes based on intent)
3. **Add multiple agent nodes** (specialist agents for different tasks)
4. **Use LangGraph's built-in memory** (MemorySaver for persistence)

---

## 📚 Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Tools Guide](https://python.langchain.com/docs/modules/tools/)
- [Groq API](https://groq.com/)

---

*Created for beginners learning LangGraph and AI agents* 🤖

