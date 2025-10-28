# simple_multi_tool_agent_groq.py
# A simple multi-tool agent example using LangChain and Groq

from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, Tool
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# Load environment variables (.env file must have GROQ_API_KEY and MODEL_NAME)
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")  # Example: llama-3.1-8b-instant

# Initialize Groq LLM
llm = ChatGroq(
    api_key=groq_api_key,
    model=MODEL_NAME,
    temperature=0.2,
    max_tokens=512
)

# --- Define Tools ---

# Tool 1: Web Search (mock)
def search_web(query: str) -> str:
    """Mock web search tool for demonstration."""
    return f"Simulated web result: '{query}' — The current weather in Nairobi is 25°C and sunny."

# Tool 2: Calculator
def calculate(expression: str) -> str:
    """A simple calculator tool for evaluating math expressions."""
    try:
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error in calculation: {e}"

# Tool 3: Summarizer
def summarize_text(text: str) -> str:
    """Summarizes a given text."""
    if len(text.split()) < 20:
        return "Text is too short to summarize."
    summary = " ".join(text.split()[:20]) + "..."
    return f"Summary: {summary}"

# Create tool objects
tools = [
    Tool(
        name="search_web",
        func=search_web,
        description="Useful for finding information or facts from the web."
    ),
    Tool(
        name="calculate",
        func=calculate,
        description="Useful for solving math expressions. Example: '5 * (3 + 2)'"
    ),
    Tool(
        name="summarize_text",
        func=summarize_text,
        description="Summarizes long passages of text into a concise form."
    )
]

# --- Define Agent Prompt ---
prompt = PromptTemplate(
    input_variables=["query"],
    template="You are a helpful AI assistant. Use tools when needed.\nQuery: {query}"
)

# --- Initialize Agent ---
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type="zero-shot-react-description",  # Enables reasoning + tool use
    verbose=True,
    max_iterations=3
)

# --- Run Agent ---
print("\n=== Agent Execution ===")

# Example queries
queries = [
    "What is the weather in Nairobi?",
    "Calculate 15 * (3 + 2)",
    "Summarize: Artificial intelligence enables machines to perform tasks that typically require human intelligence..."
]

for q in queries:
    print(f"\n--- Query: {q} ---")
    response = agent.run(q)
    print("Response:", response)

print("\n=== Done ===")
