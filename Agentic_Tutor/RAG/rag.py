import os
from crewai_tools import PDFSearchTool
from crewai_tools  import tool
from crewai import Crew
from crewai import Task
from crewai import Agent
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
        pdf='T:/Code/Project Tutor/Agentic_Tutor/RAG/{selected_pdf}',
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

# PDF Router Agent to select the appropriate PDF
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
        print("Selected PDF: class7.pdf")
        return 'class7.pdf'
    elif 'class 8' in question:
        print("Selected PDF: class8.pdf")
        return 'class8.pdf'
    elif 'class 10' in question:
        print("Selected PDF: class10.pdf")
        return 'class10.pdf'
    else:
        print("No relevant PDF found")
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

# Grader Agent to verify relevance of the retrieved content
Grader_agent = Agent(
    role='Answer Grader',
    goal='Assess relevance of the retrieved content',
    backstory="Ensure that the retrieved content is relevant to the question.",
    verbose=True,
    allow_delegation=False,
    llm=llm,
)

# Task for selecting the appropriate PDF
router_task = Task(
    description=(
        "Analyse the keywords in the question {question} to decide which PDF (class7, class8, class10) to search."
        "Return the name of the PDF as 'class7.pdf', 'class8.pdf', or 'class10.pdf'."
    ),
    expected_output="Return one of the following: 'class7.pdf', 'class8.pdf', 'class10.pdf'.",
    agent=PDF_Router_Agent,
    tools=[router_tool],
)
# Modified Retriever Agent's Task to handle missing content
retriever_task = Task(
    description=(
        "Based on the output from the router task, extract exact information from the selected PDF for the question {question}. And give to the other agent the exact info from teh pdf. "
        "If the PDF file is not available or relevant content is not found, return 'No relevant content found.'"
    ),
    expected_output="Return a clear and concise response based on the content in the selected PDF, or 'No relevant content found.' if no relevant content is available.",
    agent=Retriever_Agent,
    context=[router_task],
)

# Modify the Grader Task to check for 'No relevant content found.'
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


# Define the Crew with updated tasks
rag_crew = Crew(
    agents=[PDF_Router_Agent, Retriever_Agent, Grader_agent],
    tasks=[router_task, retriever_task, grader_task],
    verbose=True,
)

# Main execution remains the same
inputs = input("What is your question? ")
result = rag_crew.kickoff(inputs={"question": inputs})
print("\nFinal Answer:", result)