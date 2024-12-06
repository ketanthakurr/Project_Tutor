from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq
from composio_crewai import ComposioToolSet, Action, App
from datetime import datetime
import os

os.environ["GROQ_API_KEY"] = "your-api-key"

llm=ChatGroq(model="groq/llama3-8b-8192", api_key='your-api-key')

composio_toolset = ComposioToolSet()
# connected_account_id = composio_toolset.get_connected_account_id()
tools = composio_toolset.get_tools(apps = [App.GOOGLECALENDAR] )

date = datetime.today().strftime("%Y-%m-%d")

timezone = datetime.now().astimezone().tzinfo

# Define agent
gcal_create_agent = Agent(
    role="Google Calendar Create Agent",
    goal="""You take action on google calendar for creating events using google calendar api""",
    backstory=(
        """You are AI agent that is responsible for taking actions on Google calendar on users behalf. 
        You are able to create events. You need to take action on Calendar using Google Calendar API.
        Use correct tools to run apis from the given tool set."""
    ),
    verbose=True,
    tools=tools,
    llm=llm,
)

# Agent for finding events
gcal_find_agent = Agent(
    role="You take action on google calendar for finding events using google calendar api",
    goal="Search for events in Google Calendar using google calendar api.",
    backstory=(
        """You are an AI agent that helps the user search for events on Google Calendar.
        Use the search tools to locate and report event details."""
    ),
    verbose=True,
    tools=tools,
    llm=llm,
)

# Agent for updating events
gcal_update_agent = Agent(
    role="you take action on google calendar for updating events using google calendar api",
    goal="Update existing events in Google Calendar.",
    backstory=(
        """You are an AI agent that updates events in Google Calendar on behalf of the user.
        Ensure to modify events with correct parameters like time, location, and attendees."""
    ),
    verbose=True,
    tools=tools,
    llm=llm,
)

# Agent for deleting events
gcal_delete_agent = Agent(
    role="You take action on google calendar for deleting events using google calendar api",
    goal="Delete events from Google Calendar.",
    backstory=(
        """You are an AI agent that deletes events from Google Calendar on the user's behalf.
        Use the appropriate tools for the deletion process."""
    ),
    verbose=True,
    tools=tools,
    llm=llm,
)

# Create tasks for each agent
task_create = Task(
    description="Mark the event and todo in the calender using the {query}.Fetch the details like date, event name, time and description of the event form the {query}. Today's date is {date} and timezone is{timezone}. In case you need information regarding today's date it will be helpful and try to use it in case user gave words like today or tomorrow.",
    agent=gcal_create_agent,
    expected_output="Confirmation of created event",
    tools=tools
)

task_find = Task(
    description="Search for an event for the {query}. Fetch details of the event found for {query}. Today's date is {date} and timezone is {timezone}.",
    agent=gcal_find_agent,
    expected_output="Details of the event found",
    tools=tools
)

task_update = Task(
    description="Update an event for the {query}. Use the query to update the event, find details of previous event and update it according to the given {query}. Find the details like name, location, start time and end time, description, attendies gmail ids etc that needs to be updated from the {query}.Today's date is {date} and timezone is {timezone}.",
    agent=gcal_update_agent,
    expected_output="Confirmation of updated event",
    tools=tools
)

task_delete = Task(
    description="Delete an event for the {query}. Find details of event which is already scheduled in the calendar, match it and delete it according to the given {query} requirements.Today's date is {date} and timezone is {timezone}.",
    agent=gcal_delete_agent,
    expected_output="Confirmation of deleted event",
    tools=tools
)

# Combine all agents and tasks into a crew
my_crew = Crew(
    agents=[gcal_create_agent, gcal_find_agent, gcal_update_agent, gcal_delete_agent],
    tasks=[task_create, task_find, task_update, task_delete]
)

query = input("Enter the query: ")
inputs_array = [{"query": query}, {"date": date}, {"timezone": timezone}]

# Combine dictionaries into a single dictionary
inputs_dict = {k: v for d in inputs_array for k, v in d.items()}

# Kick off the tasks and print the results
results = my_crew.kickoff(inputs=inputs_dict)
print(results)