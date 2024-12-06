from crewai import Agent, Crew, Process, Task
from crewai_tools import TXTSearchTool
import os
from langchain_groq import ChatGroq

os.environ["GROQ_API_KEY"] = "your-api-key"
os.environ["COHERE_API_KEY"] = "your-api-key"

Reader_tool = TXTSearchTool(txt='T:/Code/Project Tutor/Agentic_Tutor/Summarise/notes.txt',
    config={
        "llm": {
            "provider": "groq",  # Other options include google, openai, anthropic, llama2, etc.
            "config": {
                "model": "groq/mixtral-8x7b-32768",
            },
        },
        "embedder": {
            "provider": "cohere",
        "config": {
            "model": "embed-english-v3.0",
            "api_key":os.environ["COHERE_API_KEY"],
        }
        },
    }
)

llm=ChatGroq(model="groq/gemma2-9b-it", api_key=os.environ["GROQ_API_KEY"])

quiz_generator = Agent(
    role="Teacher Agent",
    goal="Generate a quiz text given by the retriever agent.",
    verbose=True,
    backstory=(
        """The Teacher agent is responsible for generating a quiz based on the text provided by the retriever agent and if the user question is not relevant to the retrieved text, then generate it by yourself. but make sure the question is relevant to the user question."""
    ),
    expected_output="Quiz based on the text",
    llm=llm
)

retriever_agent=Agent(
    role="Retriever Agent",
    goal="Retrieve the specefic information from the extracted text according to the query.",
    verbose=True,
    backstory=(
        """The retriever agent is skilled at retrieving the specefic required text information from a large text file."""
    ),
    expected_output="Retrieved text according to query",
    tools= [Reader_tool],
    llm=llm
)

retrieve_task=Task(
    description = (
        """Based on the question {user_question} entered by the user, ensure the query is a string and use the Reader_Tool to retrieve information specific to the question from the text file. """

    ),
    expected_output="Provide exact extracted text from the whole text.",
    agent=retriever_agent,
    tools=[Reader_tool]
)

quiz_task=Task(
    description=(
        """use the retrieved text provided by the retriever agent to generate a quiz based on the topic asked by the user. {user_question}.
        Create a quiz and give the answer after each question. Ensure the questions are relevant to the topic."""
    ),
    expected_output="Quiz based on the topic and give  aclear and accurate answer to the question you asked.",
    agent=quiz_generator,
    context=[retrieve_task]
)


rag_crew = Crew(
    agents=[retriever_agent, quiz_generator],
    tasks=[retrieve_task, quiz_task],
    process=Process.hierarchical,
    planning=True,
    planning_llm=ChatGroq(model="groq/gemma2-9b-it", api_key=os.environ["GROQ_API_KEY"]),
    manager_llm=ChatGroq(model="groq/gemma2-9b-it", api_key=os.environ["GROQ_API_KEY"]),
)

user_question = input("Enter the question: ")

# Ensure query is a valid string
if not isinstance(user_question, str):
    raise ValueError("Input must be a string.")

result = rag_crew.kickoff(inputs={"user_question": user_question})
print(result)