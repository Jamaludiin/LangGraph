# create three agents with three different roles
# agent one is a researcher
# agent two is a writer
# agent three is a editor

# the researcher will research the topic
# the writer will write the content
# the editor will edit the content

# the three agents will work together to create the content 
# the content will be created in the following order:
# 1. the researcher will research the topic
# 2. the writer will write the content
# 3. the editor will edit the content

import os
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import AgentExecutor, create_react_agent
from langchain.agents.agent_toolkits import create_react_toolkit
from langchain.tools import DuckDuckGoSearchRun
from crewai import Agent, Task, Crew


# Set up the search tool
search_tool = DuckDuckGoSearchRun()

# Set up the Groq model
# Initialize the LLM
groq_llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama3-8b-8192",
    temperature=0.2,
    max_tokens=512
)

# Define the agents
researcher = Agent(
    role='Researcher',
    goal='Research the topic',
    backstory="""You are a researcher. You will research the topic and provide the information to the writer.""",
    verbose=True,
    allow_delegation=False,
    tools=[search_tool],
    llm=groq_llm
)

writer = Agent(
    role='Writer',
    goal='Write the content',
    backstory="""You are a writer. You will write the content based on the information provided by the researcher.""",
    verbose=True,
    allow_delegation=False,
    tools=[search_tool],
    llm=groq_llm
)

editor = Agent(
    role='Editor',
    goal='Edit the content',
    backstory="""You are an editor. You will edit the content based on the information provided by the writer.""",
    verbose=True,
    allow_delegation=False,
    tools=[search_tool],
    llm=groq_llm
)

# Define the tasks
task1 = Task(
    description="Research the topic",
    expected_output="The information about the topic",
    agent=researcher
)

task2 = Task(
    description="Write the content",
    expected_output="The content",
    agent=writer
)

task3 = Task(
    description="Edit the content",
    expected_output="The edited content",
    agent=editor
)

# Define the crew
crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[task1, task2, task3],
    verbose=2
)

# Run the crew
crew.kickoff()
print(crew.result)

print("###########")
print(crew.result)