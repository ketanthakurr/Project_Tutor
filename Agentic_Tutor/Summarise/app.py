import os
from gtts import gTTS
import streamlit as st
from crewai import Agent, Crew, Process, Task
from crewai_tools import TXTSearchTool
from langchain_groq import ChatGroq

# Set environment variables
os.environ["GROQ_API_KEY"] = "your-api-key"
os.environ["COHERE_API_KEY"] = "your-api-key"

# Reader tool configuration
Reader_tool = TXTSearchTool(txt='T:/Code/Project Tutor/Agentic_Tutor/Summarise/notes.txt',
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

# LLM setup
llm = ChatGroq(model="groq/gemma2-9b-it", api_key=os.environ["GROQ_API_KEY"])

# Agents setup
retriever_agent = Agent(
    role="Retriever Agent",
    goal="Retrieve specific information from the extracted text according to the query.",
    verbose=True,
    backstory="Skilled at retrieving specific required text information from a large text file.",
    expected_output="Retrieved text according to query",
    tools=[Reader_tool],
    llm=llm
)

summarise_agent = Agent(
    role="Summariser Agent",
    goal="Summarize the text given by the retriever agent.",
    verbose=True,
    backstory="Provides a concise and accurate summary of the retrieved text.",
    expected_output="Summary of the txt file",
    llm=llm
)

# question_answering_agent = Agent(
#     role="Question Answering Agent",
#     goal="Answer specific questions based on the content of the notes provided in the notes txt file.",
#     verbose=True,
#     backstory="Skilled at parsing and comprehending detailed information to deliver precise answers.",
#     expected_output="Direct answers to the questions asked based on the document's content",
#     llm=llm,
# )

# Tasks setup
retrieve_task = Task(
    description=(
        "Based on the user question, retrieve specific information from the text file using the Reader Tool."
    ),
    expected_output="Provide exact extracted text from the whole text.",
    agent=retriever_agent,
    tools=[Reader_tool]
)

summarise_task = Task(
    description=(
        "Summarize the retrieved text provided by the retriever task, ensuring it highlights key points and remains concise."
    ),
    expected_output="A clear and comprehensive summary of the text, capturing important details.",
    agent=summarise_agent,
    context=[retrieve_task]
)

# question_answer_task = Task(
#     description=(
#         "Answer the specific user question using the retrieved text from the retriever task. Add more relevant information to the response if needed."
#     ),
#     expected_output="Clear and accurate answers to the questions asked based on the document's content.",
#     agent=question_answering_agent,
#     context=[retrieve_task],
# )

# Crew setup
rag_crew = Crew(
    agents=[retriever_agent, summarise_agent],
    tasks=[retrieve_task, summarise_task],
)

# Streamlit app
st.title("AI-Powered Text Retrieval and Summarization")
st.write("Retrieve, summarize, and answer questions based on the content of a text file.")

# User input
user_question = st.text_input("Enter your question:")
if st.button("Submit"):
    if not user_question.strip():
        st.error("Please enter a valid question.")
    else:
        st.info("Processing your request...")
        with st.spinner("Retrieving and summarizing content..."):
            result = rag_crew.kickoff(inputs={"user_question": user_question})

        # Extract the actual text from the result
        if hasattr(result, 'output'):  # Check if the result has an 'output' attribute
            final_answer = result.output
        elif isinstance(result, str):
            final_answer = result  # Fallback for plain string output
        else:
            final_answer = str(result)  # Convert to string as a last resort

        st.success("Processing completed!")
        st.write("**Final Answer:**", final_answer)

        # Convert the text to speech using gTTS
        tts = gTTS(text=final_answer, lang='en')
        audio_path = "response.mp3"
        tts.save(audio_path)

        # Play the audio in Streamlit
        audio_file = open(audio_path, "rb")
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/mp3")

        # Clean up the saved audio file
        os.remove(audio_path)
