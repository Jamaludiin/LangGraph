# filename: python3 ielts_speaking_realistic_langgraph.py

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
    temperature=0.4,
    max_tokens=350
)

# ----------------------------------------------------------
# 2️⃣ Define state
# ----------------------------------------------------------
class ExamState(TypedDict):
    turn: int
    phase: str
    message: str
    topic: str

# ----------------------------------------------------------
# 3️⃣ Examiner Agent
# ----------------------------------------------------------
def examiner(state: ExamState) -> ExamState:
    turn = state["turn"]

    if turn == 1:
        prompt = """
        You are an IELTS Speaking Examiner.

        Part 1 – Introduction:
        Ask the candidate:
        - Their full name
        - Where they are from
        - Whether they work or study
        """
        phase = "Part 1"

    elif turn == 2:
        prompt = """
        You are an IELTS Speaking Examiner.

        Part 1:
        Ask about:
        - Free time activities
        - How they usually spend their weekends
        """
        phase = "Part 1"

    elif turn == 3:
        prompt = """
        You are an IELTS Speaking Examiner.

        Part 2 – Long Turn:
        Give the candidate a cue card.

        Cue Card:
        Describe a place you like to visit in your free time.
        You should say:
        - where it is
        - how often you go there
        - what you do there
        and explain why you like this place.

        Tell the candidate they have:
        - 1 minute to prepare
        - 1–2 minutes to speak
        """
        phase = "Part 2"

    else:
        prompt = f"""
        You are an IELTS Speaking Examiner.

        Part 3 – Discussion:
        Ask 2 abstract follow-up questions related to this topic:
        {state['topic']}

        Base your questions on the candidate's ideas.
        Candidate said:
        {state['message']}
        """
        phase = "Part 3"

    response = llm.invoke([HumanMessage(content=prompt)])

    print(f"\n🧑‍🏫 Examiner ({phase}, Turn {turn}):")
    print(response.content)

    return {
        "turn": turn + 1,
        "phase": phase,
        "message": response.content,
        "topic": state["topic"] or "Leisure places"
    }

# ----------------------------------------------------------
# 4️⃣ Candidate Agent
# ----------------------------------------------------------
def candidate(state: ExamState) -> ExamState:
    if state["phase"] == "Part 2":
        prompt = f"""
        You are a candidate taking the IELTS Speaking Test.

        This is Part 2 (Long Turn).
        Speak fluently for about 1–2 minutes.
        Use linking words, examples, and personal experience.

        Cue Card:
        {state['message']}
        """
    else:
        prompt = f"""
        You are a candidate taking the IELTS Speaking Test.
        Answer naturally, clearly, and confidently.

        Examiner asked:
        {state['message']}
        """

    response = llm.invoke([HumanMessage(content=prompt)])

    print(f"\n👤 Candidate (Turn {state['turn']}):")
    print(response.content)

    return {
        "turn": state["turn"],
        "phase": state["phase"],
        "message": response.content,
        "topic": state["topic"]
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
    "turn": 1,
    "phase": "",
    "message": "",
    "topic": ""
})

print("\n✅ IELTS Speaking Test Simulation Completed.")
