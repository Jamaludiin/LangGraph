# filename: single_agent_no_langGraph.py without LangGraph
# simple single agent – input -> output

from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# ----------------------------------------------------------
# 1️⃣ Load Environment Variables
# ----------------------------------------------------------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME") or "llama3-70b-8192"

# ----------------------------------------------------------
# 2️⃣ Create the Agent
# ----------------------------------------------------------
agent = ChatGroq(
    api_key=groq_api_key,
    model=MODEL_NAME,
    temperature=0.2, # temperature is a parameter that controls the randomness of the model's output
    max_tokens=512
)

# ----------------------------------------------------------
# 3️⃣ Run the Agent
# ----------------------------------------------------------
print("\n=== Running Single Agent (Input → Output) ===\n")

queries = [
    "Explain OOP in simple terms",
    "Why do students find programming difficult?",
    "What is an AI agent?"
]

for q in queries:
    print(f"🧩 User: {q}")

    response = agent.invoke(
        [HumanMessage(content=q)] # HumanMessage is a message that is sent by the user
    )

    print(f"🤖 Agent: {response.content}\n")

print("✅ Done.")
