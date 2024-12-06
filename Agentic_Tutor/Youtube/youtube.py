from crewai import Agent, Crew, Process, Task
from crewai_tools import YoutubeChannelSearchTool
import os
from langchain_groq import ChatGroq

os.environ["GROQ_API_KEY"] = "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"
os.environ["COHERE_API_KEY"] = "rQd4U691UGnTIjxMGwBIFlL4r5W5w8VeNuXJTck5"

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

llm=ChatGroq(model="groq/gemma2-9b-it", api_key=os.environ["GROQ_API_KEY"])

suggestion_agent = Agent(
    role = "Suggestion Agent",
    goal = "Suggest the best video channel to the user according to the user query.",
    verbose = True,
    backstory = (
        """
        The suggestion agent is adept in providing the best video channel to the user according to the user query.
  """),
    expected_output = "Video channel name and link",
    llm = llm
)

retriever_agent=Agent(
    role="Retriever Agent",
    # allow_delegation=True,
    goal="Retrieve the video channel information and link according to the user query.",
    verbose=True,
    backstory=(
        """The retriever agent is skilled at searching for the video channel information and link according to the user query."""
    ),
    expected_output="Retrieved youtube channel name and link according to query",
    tools= [Res_tool],
    llm = llm
)

retrieve_task=Task(
    description = (
        """Based on the question {user_question} entered by the user, use the Res_tool to retrieve information specefic to the query from the youtube channel.
        """
    ),
    expected_output="Provide exact extracted youtube channel name and link.",
    agent=retriever_agent,
    tools=[Res_tool]
)

summarise_task = Task(
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

rag_crew = Crew(
  agents=[retriever_agent, suggestion_agent],
  tasks=[retrieve_task, summarise_task],
  process=Process.hierarchical,
  planning=True,
  planning_llm=ChatGroq(model="groq/gemma2-9b-it", api_key=os.environ["GROQ_API_KEY"]),
  manager_llm=ChatGroq(model="groq/gemma2-9b-it", api_key=os.environ["GROQ_API_KEY"]),
)

user_question = input("Please enter your question: ")

result = rag_crew.kickoff(inputs={"user_question": user_question})
print(result)