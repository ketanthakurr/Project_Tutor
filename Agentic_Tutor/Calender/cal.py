from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
from composio_crewai import ComposioToolSet, Action, App
from datetime import datetime
import os

os.environ["GROQ_API_KEY"] = "your-api-key"

llm=ChatGroq(model="groq/llama3-70b-8192", api_key="your-api-key")

composio_toolset = ComposioToolSet()
# connected_account_id = composio_toolset.get_connected_account_id()
tools = composio_toolset.get_tools(apps = [App.GOOGLECALENDAR] )

date = datetime.today().strftime("%Y-%m-%d")

timezone = datetime.now().astimezone().tzinfo

# Template improvements for Google Calendar Agents

# Agent for creating events
create_event_agent = Agent(
    role="Google Calendar Event Creator",
    goal="Create new events in Google Calendar using the Google Calendar API based on user input.",
    backstory=(
        """
        You are an AI agent designed to facilitate the creation of calendar events for users. Your main task is to 
        accurately schedule new events by interacting with the Google Calendar API. Ensure the event details such as 
        title, date, time, location, and attendees are correctly captured and processed.
        Validate input data, manage potential conflicts, and respond with a clear confirmation once an event is 
        successfully created.
        """
    ),
    verbose=True,
    tools=tools,
    llm=llm,
)

# Agent for finding events
find_event_agent = Agent(
    role="Google Calendar Event Finder",
    goal="Locate and report events in Google Calendar using the Google Calendar API.",
    backstory=(
        """
        You are an AI agent specialized in searching for and retrieving event details from Google Calendar. Your role 
        includes accurately identifying events based on user queries and returning comprehensive information, including 
        event names, dates, times, locations, and participant details. Ensure precise use of search parameters and 
        respond efficiently to user requests with detailed event summaries.
        """
    ),
    verbose=True,
    tools=tools,
    llm=llm,
)

# Agent for updating events
update_event_agent = Agent(
    role="Google Calendar Event Updater",
    goal="Modify existing events in Google Calendar using the Google Calendar API based on user instructions.",
    backstory=(
        """
        You are an AI agent responsible for updating calendar events as requested by users. Your role involves 
        making targeted updates to existing events, such as changing the time, location, or participant lists. 
        Verify input details carefully and confirm with the user before finalizing updates to ensure accuracy. 
        Provide clear feedback on the status of updates once completed.
        """
    ),
    verbose=True,
    tools=tools,
    llm=llm,
)

# Agent for deleting events
delete_event_agent = Agent(
    role="Google Calendar Event Deleter",
    goal="Remove specified events from Google Calendar using the Google Calendar API.",
    backstory=(
        """
        You are an AI agent tasked with deleting events from Google Calendar as instructed by users. Your main 
        responsibility is to identify events accurately and handle their removal safely. Confirm with the user 
        before deletion and provide a clear notification when an event has been successfully removed. Handle deletion 
        requests with diligence to avoid accidental data loss.
        """
    ),
    verbose=True,
    tools=tools,
    llm=llm,
)

# Task templates for agents

task_create = Task(
    description="Create a new meeting event based on the {query}. Extract relevant details such as name, location, start and end time, description, and attendee email IDs from the {query}. Today's date is {date} and the timezone is {timezone}. Use this information especially when interpreting terms like 'today' or 'tomorrow'.",
    agent=create_event_agent,
    expected_output="Confirmation of the created event with all relevant details",
    tools=tools
)

task_find = Task(
    description="Search for an event based on the {query}. Retrieve event details such as name, date, time, location, and participants that match the {query}. Today's date is {date} and the timezone is {timezone}.",
    agent=find_event_agent,
    expected_output="Detailed report of the event found",
    tools=tools
)

task_update = Task(
    description="Update an existing event based on the {query}. Locate the event details, and apply updates such as changes in the name, location, start and end time, description, and attendee email IDs as per the {query}. Today's date is {date} and the timezone is {timezone}.",
    agent=update_event_agent,
    expected_output="Confirmation of the updated event with specifics of the changes made",
    tools=tools
)

task_delete = Task(
    description="Delete an event based on the {query}. Locate the event in the calendar, verify it matches the {query} requirements, and proceed with deletion. Today's date is {date} and the timezone is {timezone}.",
    agent=delete_event_agent,
    expected_output="Confirmation of the deleted event",
    tools=tools
)


# Combine all agents and tasks into a crew
my_crew = Crew(
    agents=[create_event_agent, find_event_agent, update_event_agent, delete_event_agent],
    tasks=[task_create, task_find, task_update, task_delete],
)


query = input("Enter the query: ")
inputs_array = [{"query": query}, {"date": date}, {"timezone": timezone}]

# Combine dictionaries into a single dictionary
inputs_dict = {k: v for d in inputs_array for k, v in d.items()}

# Kick off the tasks and print the results
results = my_crew.kickoff(inputs=inputs_dict)
print(results)