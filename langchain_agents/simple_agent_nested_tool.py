from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, Tool
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

# initialize the LLM (Groq model)
llm = ChatGroq(
    api_key=groq_api_key,
    model=MODEL_NAME,
    temperature=0.2,
    max_tokens=512
)

# define a simple tool
def search_web(query: str) -> str:
    """A mock web search tool (for demo)."""
    return f"The current weather in Nairobi is 25°C with clear skies."

# define a nested tool
def get_weather(query: str) -> str:
    """A nested tool to get the weather."""
    return search_web(query)

tool = Tool(
    name="get_weather",
    func=get_weather,
    description="Get the weather for a given location"
)

# define the agent prompt
prompt = PromptTemplate(
    input_variables=["query"],
    template="You are a helpful assistant. Use available tools if needed.\nQuestion: {query}"
)

# initialize the agent
agent = initialize_agent(
    tools=[tool],
    llm=llm,
    agent_type="react-docstore",
    verbose=True
)   

# run the agent
response = agent.run("What is the weather in Nairobi?")
print(response)