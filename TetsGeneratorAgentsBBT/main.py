import os
from dotenv import load_dotenv

from agents.workflow import app
from agents.AgentState import AgentState # Import AgentState for type hinting

# ----------------------------------------------------------
# 1️⃣ Load environment variables
# ----------------------------------------------------------
load_dotenv()
# groq_api_key is loaded in environmentVariables.py
# MODEL_NAME is loaded in environmentVariables.py

# ----------------------------------------------------------
# 2️⃣ Main execution logic
# ----------------------------------------------------------
if __name__ == "__main__":
    # Read the code from the put.py file
    with open("/Users/LLM and HuggingFace/LangGraph/TetsGeneratorAgentsBBT/programUnderTest/put.py", "r") as file:
        code_under_test = file.read()

    # Invoke the workflow with the initial state
    result = app.invoke({"code_under_test": code_under_test})
    
    print("\n=== FINAL OUTPUT ===\n")
    print(result["test_case"])

    # Write the test case to a file
    output_file_path = "/Users/LLM and HuggingFace/LangGraph/TetsGeneratorAgentsBBT/generatedTest/test_case.py"
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, "w") as file:
        file.write(result["test_case"])
