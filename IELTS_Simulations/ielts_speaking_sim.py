# filename: python3 ielts_speaking_sim.py
# working correctly
from typing import TypedDict
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

# ----------------------------------------------------------
# 1️⃣ Load environment
# ----------------------------------------------------------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME") or "llama3-70b-8192"

llm = ChatGroq(
    api_key=groq_api_key,
    model=MODEL_NAME,
    temperature=0.3,
    max_tokens=200
)

# ----------------------------------------------------------
# 2️⃣ Define state
# ----------------------------------------------------------
class ExamState(TypedDict):
    turn: int
    message: str

# ----------------------------------------------------------
# 3️⃣ Agent One (Examiner)
# ----------------------------------------------------------
def examiner(state: ExamState) -> ExamState:
    # IELTS speaking procedure based on turn
    if state["turn"] == 1:
                prompt = """
                You are an IELTS Speaking Examiner.
                Ask the candidate a few introductory questions (Part 1):
                - Name
                - Hometown
                - Studies or Work
                """
    elif state["turn"] == 2:
                prompt = """
                You are an IELTS Speaking Examiner.
                Ask the candidate about hobbies and daily routines (Part 1).
                """
    elif state["turn"] == 3:
                prompt = """
                You are an IELTS Speaking Examiner.
                Give the candidate a Part 2 speaking topic.
                Ask them to speak for 1-2 minutes on the topic.
                """
    else:
                prompt = f"""
                You are an IELTS Speaking Examiner.
                Ask Part 3 discussion questions based on previous candidate answers.
                Candidate said:
                {state['message']}
                """

    response = llm.invoke([HumanMessage(content=prompt)])

    print(f"\n🧑‍🏫 Examiner (Turn {state['turn']}):")
    print(response.content)

    return {
        "message": response.content,
        "turn": state["turn"] + 1
    }

# ----------------------------------------------------------
# 4️⃣ Agent Two (Candidate)
# ----------------------------------------------------------
def candidate(state: ExamState) -> ExamState:
    response = llm.invoke([HumanMessage(
        content=f"""
                You are a candidate taking the IELTS Speaking Test.
                Answer the examiner's questions naturally and briefly.

                Examiner asked:
                {state['message']}
                """
    )])

    print(f"\n👤 Candidate (Turn {state['turn']}):")
    print(response.content)

    return {
        "message": response.content,
        "turn": state["turn"]
    }

# ----------------------------------------------------------
# 5️⃣ Loop condition
# ----------------------------------------------------------
def should_continue(state: ExamState):
    if state["turn"] > 5:
        return END
    return "examiner"

# ----------------------------------------------------------
# 6️⃣ Build LangGraph
# ----------------------------------------------------------
graph = StateGraph(ExamState)
graph.add_node("examiner", examiner)
graph.add_node("candidate", candidate)

graph.set_entry_point("examiner")
graph.add_edge("examiner", "candidate")
graph.add_conditional_edges("candidate", should_continue)

app = graph.compile()

# ----------------------------------------------------------
# 7️⃣ Run
# ----------------------------------------------------------
app.invoke({
    "message": "",
    "turn": 1
})

print("\n✅ IELTS Simulation Done.")
