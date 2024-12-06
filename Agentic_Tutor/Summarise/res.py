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

summarise_agent = Agent(
    role = "Summariser Agent",
    goal = "Summarize the text given by the retriever agent.",
    verbose = True,
    backstory = (
        """
        The summarise agent is adept in providing an abstractive summary of the notes and the content given by the retriever agent taking in count important details like facts, figures, concept etc and ensuring that the summary is concise and short.
  """),
    expected_output = "Summary of the txt file",
    llm = llm
)

retriever_agent=Agent(
    role="Retriever Agent",
    # allow_delegation=True,
    goal="Retrieve the specefic information from the extracted text according to the query.",
    verbose=True,
    backstory=(
        """The retriever agent is skilled at retrieving the specefic required text information from a large text file."""
    ),
    expected_output="Retrieved text according to query",
    tools= [Reader_tool],
    llm = llm
)

question_answering_agent = Agent(
    role = "Question Answering Agent",
    goal = "Answer specific questions based on the content of the notes provided in the notes txt file.",
    # allow_delegation= True,
    verbose = True,
    backstory = (
        """
        The question answering agent is skilled at parsing and comprehending detailed information from provided documents to deliver precise answers. It can identify relevant sections within the text and return concise, relevant responses.
        """
    ),
    expected_output = "Direct answers to the questions asked based on the document's content",
    # tools= [Reader_tool],
    llm = llm,
)

retrieve_task=Task(
    description = (
        """Based on the question {user_question} entered by the user, use the Reader_Tool to retrieve information specefic to the question from the text file.
        """
    ),
    expected_output="Provide exact extracted text from the whole text.",
    agent=retriever_agent,
    tools=[Reader_tool]
)

summarise_task = Task(
    description = (
        """
        Use the retrieved text provided by the retrieve_task to provide summary to the question asked by the user.{user_question}.
        Summarize the retrieved text, highlighting key points such as facts, figures, concepts, etc.
        Ensure the summary is concise and encompasses all essential details.
        """
    ),
    expected_output=
    """
    A clear and comprehensive summary of the text, capturing important details.
    """,
    # tools=[Reader_tool],
    agent=summarise_agent,
    context=[retrieve_task]
)

question_answer_task = Task(
    description = (
        """
        Use the retrieved text provided by the retrieve_task to answer the specefic question of the user {user_question}. Add more information to the retrieved text.
        Your final answer MUST be clear and accurate, based on the content of the extracted text as well as your knowledge.
        """
    ),
    expected_output=
    """
    Provide clear and accurate answers to the questions asked based on the document's content.
    """,
    # tools=[Reader_tool] ,
    agent=question_answering_agent,
    context=[retrieve_task],
    # human_input=True
)

rag_crew = Crew(
  agents=[retriever_agent,question_answering_agent, summarise_agent],
  tasks=[retrieve_task,question_answer_task, summarise_task],
  process=Process.hierarchical,
  planning=True,
  planning_llm=ChatGroq(model="groq/gemma2-9b-it", api_key=os.environ["GROQ_API_KEY"]),
  manager_llm=ChatGroq(model="groq/gemma2-9b-it", api_key=os.environ["GROQ_API_KEY"]),
#   memory=True,
)


user_question = input("Please enter your question: ")

result = rag_crew.kickoff(inputs={"user_question": user_question})
print(result)