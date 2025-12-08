import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
# ----------------------------------------------------------
# 1️⃣ Load environment variables
# ----------------------------------------------------------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME") or "llama3-70b-8192"



# ----------------------------------------------------------
# 3️⃣ LLM + Tools
# ----------------------------------------------------------
llm = ChatGroq(
    api_key=groq_api_key,
    model=MODEL_NAME,
    temperature=0.2,
)