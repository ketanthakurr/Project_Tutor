import os
import streamlit as st
from crewai_tools import PDFSearchTool
from crewai_tools import tool
from crewai import Crew, Task, Agent
from langchain_groq import ChatGroq

# Set up API key for Groq
os.environ["GROQ_API_KEY"] = "your-api-key"

# LLM setup using Groq API
llm = ChatGroq(
    model="groq/mixtral-8x7b-32768",
    api_key=os.environ["GROQ_API_KEY"]
)

# Function to dynamically configure the PDFSearchTool
def get_rag_tool(selected_pdf):
    return PDFSearchTool(
        pdf=f'T:/Code/Project Tutor/Agentic_Tutor/RAG/{selected_pdf}',
        config=dict(
            llm=dict(
                provider="groq",
                config=dict(
                    model="groq/mixtral-8x7b-32768",
                ),
            ),
            embedder=dict(
                provider="huggingface",
                config=dict(
                    model="BAAI/bge-small-en-v1.5",
                ),
            ),
        )
    )

# PDF Router Agent
PDF_Router_Agent = Agent(
    role="PDF Selector",
    goal="Select the relevant PDF based on the class mentioned in the question.",
    backstory=(
        "You are responsible for selecting the correct PDF document to search based on the class specified in the question."
        "If the question is about class 7, use 'class7.pdf'; if it's about class 8, use 'class8.pdf'; "
        "and if it's about class 10, use 'class10.pdf'."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm,
)

@tool
def router_tool(question):
    """Router Function to select PDF"""
    if 'class 7' in question:
        return 'class7.pdf'
    elif 'class 8' in question:
        return 'class8.pdf'
    elif 'class 10' in question:
        return 'class10.pdf'
    else:
        return None

# Retriever Agent
Retriever_Agent = Agent(
    role="Retriever",
    goal="Use the selected PDF to answer the question",
    backstory="Retrieve relevant information from the PDF to answer the user's question.",
    verbose=True,
    allow_delegation=False,
    llm=llm,
)

# Grader Agent
Grader_agent = Agent(
    role='Answer Grader',
    goal='Assess relevance of the retrieved content',
    backstory="Ensure that the retrieved content is relevant to the question.",
    verbose=True,
    allow_delegation=False,
    llm=llm,
)

# Router Task
router_task = Task(
    description=(
        "Analyze the keywords in the question {question} to decide which PDF (class7, class8, class10) to search."
        "Return the name of the PDF as 'class7.pdf', 'class8.pdf', or 'class10.pdf'."
    ),
    expected_output="Return one of the following: 'class7.pdf', 'class8.pdf', 'class10.pdf'.",
    agent=PDF_Router_Agent,
    tools=[router_tool],
)

# Retriever Task
retriever_task = Task(
    description=(
        "Based on the output from the router task, extract exact information from the selected PDF for the question {question}. "
        "If the PDF file is not available or relevant content is not found, return 'No relevant content found.'"
    ),
    expected_output="Return a clear and concise response based on the content in the selected PDF, or 'No relevant content found.' if no relevant content is available.",
    agent=Retriever_Agent,
    context=[router_task],
)

# Grader Task
grader_task = Task(
    description=(
        "Evaluate whether the retrieved content is relevant to the question {question}. "
        "If the retrieved content is 'No relevant content found.', generate a fresh answer only if it’s general knowledge. "
        "Otherwise, state 'Content not found in the provided PDF.'"
    ),
    expected_output="Respond 'yes' if relevant, or 'no' if not. Provide a new answer only if unrelated.",
    agent=Grader_agent,
    context=[retriever_task],
)

# Define the Crew
rag_crew = Crew(
    agents=[PDF_Router_Agent, Retriever_Agent, Grader_agent],
    tasks=[router_task, retriever_task, grader_task],
    verbose=True,
)

# Streamlit UI
st.title("PDF-Based Question Answering System")
st.write("Retrieve information from a specific class-based PDF.")

user_question = st.text_input("Enter your question:")

if st.button("Submit"):
    if not user_question.strip():
        st.error("Please enter a valid question.")
    else:
        with st.spinner("Processing..."):
            result = rag_crew.kickoff(inputs={"question": user_question})
        st.success("Answer Retrieved!")
        st.write(result)
