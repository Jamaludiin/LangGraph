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
def search_web(query: str) -> str:
    """Mock web search tool for demonstration."""
    return f"Simulated web result: '{query}' — The current weather in Nairobi is 25°C and sunny."

def calculate(expression: str) -> str:
    """Simple calculator."""
    try:
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error in calculation: {e}"

def summarize_text(text: str) -> str:
    """Summarizes a given text."""
    if len(text.split()) < 20:
        return "Text is too short to summarize."
    summary = " ".join(text.split()[:20]) + "..."
    return f"Summary: {summary}"

# Create tools
tools = [
    Tool(name="search_web", func=search_web, description="Useful for web queries."),
    Tool(name="calculate", func=calculate, description="Useful for math operations."),
    Tool(name="summarize_text", func=summarize_text, description="Useful for summarizing long text."),
]

# --- Initialize Agent ---
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type="zero-shot-react-description",
    verbose=False,
    max_iterations=3  # Allow more thinking steps
)

# --- Intelligent Query Handling ---
queries = [
    "What is the weather in Nairobi?",
    "Calculate 15 * (3 + 2)",
    "Summarize: Artificial intelligence enables machines to perform tasks that typically require human intelligence..."
]

for q in queries:
    print(f"\n--- Query: {q} ---")
    #response = agent.invoke({"input": q})   # ✅ use invoke instead of run
    response = agent.invoke({"input": q})
    print("Final Answer:", response["output"])
    
    
