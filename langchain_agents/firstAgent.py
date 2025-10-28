#create simple agent that can be used to test the langchain framework
# simple_agent_groq.py
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, Tool
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# Load API key from .env file
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

# Initialize the LLM (Groq model)
llm = ChatGroq(
    api_key=groq_api_key,
    model=MODEL_NAME,  # You can change to another Groq model if needed
    temperature=0.2,
    max_tokens=512
)

# Define a simple tool
def search_web(query: str) -> str:
    """A mock web search tool (for demo)."""
    # You could later integrate with an API like DuckDuckGo or SerpAPI.
    return f"Simulated web search result for: {query}"

tool = Tool(
    name="search_web",
    func=search_web,
    description="Search the web for given information"
)

# Define the agent prompt
prompt = PromptTemplate(
    input_variables=["query"],
    template="You are a helpful assistant. Use available tools if needed.\nQuestion: {query}"
)

# Create a simple agent using the tool and Groq model
agent = initialize_agent(
    tools=[tool],
    llm=llm,
    agent_type="zero-shot-react-description",
    verbose=True
)

# Run the agent
response = agent.run("What is the weather in Nairobi?")
print("\n=== Agent Response ===")
print(response)
