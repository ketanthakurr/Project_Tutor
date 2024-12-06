import os
from crewai import Crew
from crewai import Process
from crewai import Task
from crewai import Agent
from datetime import datetime
from crewai_tools  import tool
from composio_crewai import App
from composio_crewai import Action
from langchain_groq import ChatGroq
from crewai_tools import PDFSearchTool
from crewai_tools import TXTSearchTool
from composio_crewai import ComposioToolSet
from crewai_tools import YoutubeChannelSearchTool


os.environ["GROQ_API_KEY"] = "your-api-key"
os.environ["COHERE_API_KEY"] = "your-api-key"

# Summaries llm
llm_summaries=ChatGroq(model="groq/gemma2-9b-it", api_key=os.environ["GROQ_API_KEY"])
llm=ChatGroq(model="groq/gemma2-9b-it", api_key=os.environ["GROQ_API_KEY"])

# Calender llm
llm_calender=ChatGroq(model="groq/llama3-8b-8192", api_key=os.environ["GROQ_API_KEY"])


# ---------------------------------------------------Composio----------------------------------------------------
composio_toolset = ComposioToolSet()

tools = composio_toolset.get_tools(apps = [App.GOOGLECALENDAR] )

date = datetime.today().strftime("%Y-%m-%d")

timezone = datetime.now().astimezone().tzinfo

# -----------------------------------------------------TOOLS--------------------------------------------------

Reader_tool = TXTSearchTool(txt='T:/Code/Project Tutor/Agentic_Tutor/notes.txt',
    config={
        "llm": {
            "provider": "groq",
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

Reader_quiz_tool = TXTSearchTool(txt='T:/Code/Project Tutor/Agentic_Tutor/Summarise/notes.txt',
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


Res_tool = YoutubeChannelSearchTool(
    config={
        "llm": {
            "provider": "groq",
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






# -----------------------------------------------------Summariser Agents--------------------------------------------------

summarise_agent = Agent(
    role = "Summariser Agent",
    goal = "Summarize the text given by the retriever agent.",
    verbose = True,
    backstory = (
        """
        The summarise agent is adept in providing an abstractive summary of the notes and the content given by the retriever agent taking in count important details like facts, figures, concept etc and ensuring that the summary is concise and short.
        """),
    expected_output = "Summary of the txt file",
    llm = llm_summaries,
    allow_delegation=False,
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
    llm = llm_summaries,
    allow_delegation=False,
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
    llm = llm_summaries,
    allow_delegation=False,
)

# ---------------------------------------Quiz Generator--------------------------------------------------

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

retriever_quiz_agent=Agent(
    role="Retriever Agent",
    goal="Retrieve the specefic information from the extracted text according to the query.",
    verbose=True,
    backstory=(
        """The retriever agent is skilled at retrieving the specefic required text information from a large text file."""
    ),
    expected_output="Retrieved text according to query",
    tools= [Reader_quiz_tool],
    llm=llm
)

# ----------------------------------------------------Youtube Agents--------------------------------------------------

suggestion_agent = Agent(
    role = "Suggestion Agent",
    goal = "Suggest the best video channel to the user according to the user query.",
    verbose = True,
    backstory = (
        """
        The suggestion agent is adept in providing the best video channel to the user according to the user query.
  """),
    expected_output = "Video channel name and link",
    llm = llm,
    allow_delegation=False,
)

retriever_yt_agent=Agent(
    role="Retriever youtube Agent",
    # allow_delegation=True,
    goal="Retrieve the video channel information and link according to the user query.",
    verbose=True,
    backstory=(
        """The retriever agent is skilled at searching for the video channel information and link according to the user query."""
    ),
    expected_output="Retrieved youtube channel name and link according to query",
    tools= [Res_tool],
    llm = llm,
    allow_delegation=False,
)


# -----------------------------------------------------Summerise Tasks--------------------------------------------------

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

# --------------------------------------Quiz Tasks--------------------------------------------------


retrieve_quiz_task=Task(
    description = (
        """Based on the question {user_question} entered by the user, ensure the query is a string and use the Reader_Tool to retrieve information specific to the question from the text file. """

    ),
    expected_output="Provide exact extracted text from the whole text.",
    agent=retriever_quiz_agent,
    tools=[Reader_tool]
)

quiz_task=Task(
    description=(
        """use the retrieved text provided by the retriever agent to generate a quiz based on the topic asked by the user. {user_question}.
        Create a quiz and give the answer after each question. Ensure the questions are relevant to the topic."""
    ),
    expected_output="Quiz based on the topic and give  aclear and accurate answer to the question you asked.",
    agent=quiz_generator,
    context=[retrieve_quiz_task]
)

# -----------------------------------------------------Youtube Tasks--------------------------------------------------

retrieve_yt_task=Task(
    description = (
        """Based on the question {user_question} entered by the user, use the Res_tool to retrieve information specefic to the query from the youtube channel.
        """
    ),
    expected_output="Provide exact extracted youtube channel name and link.",
    agent=retriever_yt_agent,
    tools=[Res_tool]
)

summarise_yt_task = Task(
    description = (
        """
        Use the retrieved text provided by the retrieve_task to provide the name of the video channel asked by the user.{user_question}.
        Write about the youtube channel and which video should he watch accoridng ot the query.
        Ensure the youtube name and link is clear.
        """
    ),
    expected_output=
    """
    Youtube channel name and link with the video to watch.
    """,
    # tools=[Reader_tool],
    agent=suggestion_agent,
    context=[retrieve_task]
)


# Manager Agent
manager = Agent(
    role="Project Manager",
    goal="Efficiently manage the crew and delegate tasks based on user queries.",
    verbose=True,
    backstory=(
        """
        You are a highly efficient project manager responsible for coordinating various AI agents to fulfill user requests.
        Your job is to understand the user's query and delegate the appropriate tasks to either the Youtube agents (appropriate suggestion_agent and retrieve_yt_agent.) or the summarization agents (retrieve_task, summarise_task, and question_answer_task) and if the user asked for the quiz delegate it to the quiz_generator agent, retriever agent.
        You need to identify whether the query is related to Google Calendar events or requires text summarization/answering.
        """
    ),
    allow_delegation=True,
    llm=llm,
)

# Manager Task
manager_task = Task(
    description=(
        """
        Based on the user query {user_question}, determine if the query is related to Youtube or if it requires summarization or question answering.
        - If the query involves keywords like 'Youtube', 'suggestion', 'channel', 'youtube', 'video', assign the task to the appropriate suggestion_agent and retrieve_yt_agent.
        - If the query involves keywords like 'summarize', 'notes', 'answer', or 'words', delegate it to the retrieve_task, summarise_task, and question_answer_task.
        - If the query involves keywords like 'quiz', 'generate', 'question', or 'mcq', delegate it to the retrieve_task and quiz_task.
        - And make sure the other agents dont interupt the process and should not be assigned the task.
        """
    ),
    agent=manager,
    expected_output="Assigns the task to the relevant agents based on the user's query."
)


crew = Crew(
    agents=[summarise_agent,retriever_agent,question_answering_agent, suggestion_agent, retriever_yt_agent, quiz_generator, retriever_quiz_agent],
    tasks=[retrieve_task, summarise_task, question_answer_task, retrieve_yt_task, summarise_yt_task,retrieve_quiz_task, quiz_task, manager_task],
    manager_agent=manager,
    process=Process.hierarchical,
    # planning=True,
    # planning_llm=ChatGroq(model="groq/gemma2-9b-it", api_key=os.environ["GROQ_API_KEY"]),
    # manager_llm=ChatGroq(model="groq/gemma2-9b-it", api_key=os.environ["GROQ_API_KEY"]),

)

user_question = input("Please enter your question: ")

result = crew.kickoff(inputs={"user_question": user_question})
print(result)



