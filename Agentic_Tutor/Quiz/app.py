import streamlit as st
from crewai import Agent, Crew, Process, Task
from crewai_tools import TXTSearchTool
import os
from langchain_groq import ChatGroq

# Set environment variables
os.environ["GROQ_API_KEY"] = "your-api-key"
os.environ["COHERE_API_KEY"] = "your-api-key"

# Define Reader Tool
Reader_tool = TXTSearchTool(
    txt='T:/Code/Project Tutor/Agentic_Tutor/Summarise/notes.txt',
    config={
        "llm": {
            "provider": "groq",
            "config": {"model": "groq/mixtral-8x7b-32768"},
        },
        "embedder": {
            "provider": "cohere",
            "config": {
                "model": "embed-english-v3.0",
                "api_key": os.environ["COHERE_API_KEY"],
            },
        },
    },
)

# Initialize LLM
llm = ChatGroq(model="groq/gemma2-9b-it", api_key=os.environ["GROQ_API_KEY"])

# Define Agents
quiz_generator = Agent(
    role="Teacher Agent",
    goal="Generate a quiz text given by the retriever agent.",
    verbose=True,
    backstory=(
        """The Teacher agent is responsible for generating a quiz based on the text provided by the retriever agent. If the user question is not relevant to the retrieved text, then generate it yourself. Ensure the questions are relevant to the user's query."""
    ),
    expected_output="Quiz based on the text",
    llm=llm,
)

retriever_agent = Agent(
    role="Retriever Agent",
    goal="Retrieve specific information from the extracted text according to the query.",
    verbose=True,
    backstory=(
        """The retriever agent is skilled at retrieving specific required text information from a large text file."""
    ),
    expected_output="Retrieved text according to query",
    tools=[Reader_tool],
    llm=llm,
)

# Define Tasks
retrieve_task = Task(
    description=(
        """Based on the question {user_question} entered by the user, ensure the query is a string and use the Reader_Tool to retrieve information specific to the question from the text file."""
    ),
    expected_output="Provide exact extracted text from the whole text.",
    agent=retriever_agent,
    tools=[Reader_tool],
)

quiz_task = Task(
    description=(
        """Use the retrieved text provided by the retriever agent to generate a quiz based on the topic asked by the user: {user_question}.
        Create a quiz and give the answer after each question. Ensure the questions are relevant to the topic."""
    ),
    expected_output="Quiz based on the topic with clear and accurate answers.",
    agent=quiz_generator,
    context=[retrieve_task],
)

# Define Crew
rag_crew = Crew(
    agents=[retriever_agent, quiz_generator],
    tasks=[retrieve_task, quiz_task],
    process=Process.hierarchical,
    planning=True,
    planning_llm=ChatGroq(model="groq/gemma2-9b-it", api_key=os.environ["GROQ_API_KEY"]),
    manager_llm=ChatGroq(model="groq/gemma2-9b-it", api_key=os.environ["GROQ_API_KEY"]),
)

# Streamlit UI
st.title("Quiz Generator App")
st.write("Retrieve information from a text file and generate a quiz.")

user_question = st.text_input("Enter your question:")

if st.button("Generate Quiz"):
    if not isinstance(user_question, str) or not user_question.strip():
        st.error("Please enter a valid question.")
    else:
        with st.spinner("Processing..."):
            result = rag_crew.kickoff(inputs={"user_question": user_question})
        st.success("Quiz Generated!")
        st.write(result)
