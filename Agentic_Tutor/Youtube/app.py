import streamlit as st
from crewai import Agent, Crew, Process, Task
from crewai_tools import YoutubeChannelSearchTool
import os
from langchain_groq import ChatGroq

# Environment Variables
os.environ["GROQ_API_KEY"] = "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"
os.environ["COHERE_API_KEY"] = "rQd4U691UGnTIjxMGwBIFlL4r5W5w8VeNuXJTck5"

# Tool for retrieving YouTube channels
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
                "api_key": os.environ["COHERE_API_KEY"],
            }
        },
    }
)

# Language model
llm = ChatGroq(model="groq/gemma2-9b-it", api_key=os.environ["GROQ_API_KEY"])

# Agents
suggestion_agent = Agent(
    role="Suggestion Agent",
    goal="Suggest the best video channel to the user according to the user query.",
    verbose=True,
    backstory="The suggestion agent provides the best video channel according to the user query.",
    expected_output="Video channel name and link",
    llm=llm
)

retriever_agent = Agent(
    role="Retriever Agent",
    goal="Retrieve the video channel information and link according to the user query.",
    verbose=True,
    backstory="The retriever agent searches for video channel information and links for the user query.",
    expected_output="Retrieved YouTube channel name and link according to query",
    tools=[Res_tool],
    llm=llm
)

# Tasks
retrieve_task = Task(
    description=(
        "Based on the question {user_question} entered by the user, "
        "use the Res_tool to retrieve specific YouTube channel information."
    ),
    expected_output="Provide exact extracted YouTube channel name and link.",
    agent=retriever_agent,
    tools=[Res_tool]
)

summarise_task = Task(
    description=(
        "Use the retrieved text provided by the retrieve_task to provide the name "
        "of the video channel asked by the user {user_question}. Write about the "
        "YouTube channel and which video should they watch according to the query. "
        "Ensure the YouTube name and link is clear."
    ),
    expected_output="YouTube channel name and link with the video to watch.",
    agent=suggestion_agent,
    context=[retrieve_task]
)

# RAG Crew
rag_crew = Crew(
    agents=[retriever_agent, suggestion_agent],
    tasks=[retrieve_task, summarise_task],
    process=Process.hierarchical,
    planning=True,
    planning_llm=ChatGroq(model="groq/gemma2-9b-it", api_key=os.environ["GROQ_API_KEY"]),
    manager_llm=ChatGroq(model="groq/gemma2-9b-it", api_key=os.environ["GROQ_API_KEY"]),
)

# Streamlit UI
st.title("YouTube Channel Suggestion Tool")
st.write("Ask any question to find relevant YouTube channels and videos.")

user_question = st.text_input("Enter your query:")

if st.button("Submit"):
    if not user_question.strip():
        st.error("Please enter a valid question.")
    else:
        st.info("Processing your request...")
        with st.spinner("Searching YouTube channels..."):
            result = rag_crew.kickoff(inputs={"user_question": user_question})
        
        # Extract and display results
        if hasattr(result, 'output'):
            final_result = result.output
        elif isinstance(result, str):
            final_result = result
        else:
            final_result = str(result)

        st.success("Here are your results!")
        st.write("**Suggested Channel and Video:**", final_result)
