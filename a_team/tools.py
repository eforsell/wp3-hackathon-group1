from langchain.tools import tool
from pydantic import BaseModel, Field
from datetime import datetime
import schedule
from pathlib import Path
import pandas as pd
from ics import Calendar
from colorama import Fore, Style

current_file = Path(__file__)


class CreateEventArgsSchema(BaseModel):
    employee_ids: list = Field(
        description="The ids of the employees whose calendar you want to update"
    )
    title: str = Field(description="The title of the calendar event")
    start_date: datetime = Field(
        description="The start date and time of the calendar event"
    )
    end_date: datetime = Field(
        description="The end date and time of the calendar event"
    )
    description: str = Field(description="The event description")
    location: str = Field(
        description="Make up the location for the event, e.g. a meeting room."
    )


@tool(args_schema=CreateEventArgsSchema)
def create_calendar_event(employee_ids: list, *args, **kwargs):
    """Use this to create a calendar event."""
    print(Fore.RED + "create_calendar_event called" + Style.RESET_ALL)
    event = schedule.create.ics_event(*args, **kwargs)
    for employee_id in employee_ids:
        with open(
            f"{current_file.parent}/schedule/updated_calendars/{employee_id}.ics", "a"
        ) as f:
            f.write(f"{event}\n\n")
    return f"Created event: {event}"


@tool()
def get_persons():
    """Use this to get all employees, including their name and roles as well as id."""
    print(Fore.RED + "get_persons called" + Style.RESET_ALL)
    return pd.read_csv(
        f"{current_file.parent}/data/employees.csv", sep=";", index_col=[0]
    )


@tool()
def get_calendars(ids: list):
    """Use this to get the calendar for each employee by their id."""
    print(Fore.RED + "get_calendars called" + Style.RESET_ALL)
    calendars = {}
    for employee_id in ids:
        with open(
            f"{current_file.parent}/schedule/calendars/{employee_id}.ics",
            "r",
            encoding="utf-8",
        ) as f:
            cal = Calendar.parse_multiple(f.read())
            calendars[employee_id] = [str(event) for event in cal]
    return calendars
