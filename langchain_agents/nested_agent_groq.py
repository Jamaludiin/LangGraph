# nested_agent_groq.py
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, Tool
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

# Initialize Groq LLM
llm = ChatGroq(
    api_key=groq_api_key,
    model=MODEL_NAME,
    temperature=0.2,
    max_tokens=512
)

# --- TOOL 1: Search the web ---
def search_web(query: str) -> str:
    """Mock search tool."""
    if "weather" in query.lower():
        return "Today's forecast shows 25°C and sunny skies in Nairobi."
    elif "population" in query.lower():
        return "Nairobi has a population of around 4.3 million people."
    else:
        return f"Search results for '{query}' are unavailable."

# --- TOOL 2: Analyze weather (depends on TOOL 1) ---
def analyze_weather(city: str) -> str:
    """A dependent tool that calls the search_web tool."""
    result = search_web(f"weather in {city}")
    # The tool processes or extracts details from the first tool
    if "25" in result:
        return f"{city} seems warm and sunny — 25°C! Perfect day to go out."
    else:
        return f"Couldn't determine the exact weather for {city}."

# Create Tool objects
tools = [
    Tool(
        name="SearchWeb",
        func=search_web,
        description="Search the web for general information or weather data."
    ),
    Tool(
        name="AnalyzeWeather",
        func=analyze_weather,
        description="Analyze weather for a city using SearchWeb results."
    ),
]

# Initialize agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type="zero-shot-react-description",
    verbose=True,
    max_iterations=3
)

# Run the agent
response = agent.run("Please analyze the weather in Nairobi.")
print("\n=== Agent Response ===")
print(response)
