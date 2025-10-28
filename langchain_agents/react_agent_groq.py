# filename: react_agent_groq.py

from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# ----------------------------------------------------------
# Load Environment Variables
# ----------------------------------------------------------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME") or "llama-3.1-8b-instant"

# ----------------------------------------------------------
# Initialize the LLM
# ----------------------------------------------------------
llm = ChatGroq(
    api_key=groq_api_key,
    model=MODEL_NAME,
    temperature=0.2,
    max_tokens=512
)

# ----------------------------------------------------------
# Define Tools
# ----------------------------------------------------------
@tool
def search_web(query: str) -> str:
    """Search the web for information or weather updates."""
    if "weather" in query.lower():
        return "The current weather in Nairobi is 25°C and sunny."
    elif "population" in query.lower():
        return "Nairobi has a population of about 4.3 million."
    else:
        return f"No data found for '{query}'."

@tool
def calculate(expression: str) -> str:
    """Perform basic math calculations."""
    try:
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error calculating expression: {e}"

@tool
def summarize_text(text: str) -> str:
    """Summarize the given text into a short version."""
    if len(text.split()) < 15:
        return "Text too short to summarize."
    summary = " ".join(text.split()[:15]) + "..."
    return f"Summary: {summary}"


tools = [search_web, calculate, summarize_text]

# ----------------------------------------------------------
# Define Agent Prompt (includes 'agent_scratchpad'!)
# ----------------------------------------------------------
prompt = PromptTemplate(
    input_variables=["input", "agent_scratchpad"],
    template=(
        "You are a helpful and intelligent AI assistant. "
        "You have access to several tools to help answer questions.\n\n"
        "If necessary, use these tools to gather information before answering.\n\n"
        "Tools available: search_web, calculate, summarize_text\n\n"
        "User Query: {input}\n\n"
        "Previous tool results (if any):\n{agent_scratchpad}\n\n"
        "Now provide the best final answer to the user."
    ),
)

# ----------------------------------------------------------
# Create Agent + Executor
# ----------------------------------------------------------
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# ----------------------------------------------------------
# Run Example Queries
# ----------------------------------------------------------
queries = [
    "What is the weather in Nairobi?",
    "Calculate 15 * (3 + 2)",
    "Summarize: Artificial intelligence enables machines to perform tasks that typically require human intelligence such as reasoning, learning, and problem solving."
]

print("\n================= Running Agent =================\n")
for q in queries:
    print(f"🧩 Query: {q}")
    try:
        result = agent_executor.invoke({"input": q})
        print("✅ Final Answer:", result["output"], "\n")
    except Exception as e:
        print("❌ Error:", e, "\n")





# ----------------------------------------------------------
# Ask a Specific Question
# ----------------------------------------------------------
user_question = input("🧠 Ask your question: ")

result = agent_executor.invoke({"input": user_question})
print("\n✅ Final Answer:", result["output"])


"""
🧩 What is agent_scratchpad?

agent_scratchpad is a special variable used internally by LangChain agents to keep track of the agent’s thought process and previous tool interactions during multi-step reasoning.

Think of it as the agent’s notebook or memory where it writes:

What tools it called

What those tools returned

What it has learned so far

What it plans to do next
"""