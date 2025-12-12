from ics import Event
from datetime import datetime
import pytz


def ics_event(
    title: str,
    start_date: datetime,
    end_date: datetime,
    description: str | None = None,
    location: str | None = None,
    timezone: str = "UTC",
) -> Event:
    """
    Create an ICS calendar event file.

    Example:

        ics_event(
            title="Team Meeting",
            start_date=datetime(2025, 12, 15, 14),
            end_date=datetime(2025, 12, 15, 16),
            description="Monthly project update meeting.",
            location="Conference Room A",
        )

    :param title: Event title (string)
    :param start_date: Start datetime in 'YYYY-MM-DD HH:MM' format
    :param end_date: End datetime in 'YYYY-MM-DD HH:MM' format
    :param description: Event description (string)
    :param location: Event location (string)
    """
    try:
        # Validate and parse datetime strings
        tz = pytz.timezone(timezone)
        start_dt = tz.localize(start_date)
        end_dt = tz.localize(end_date)

        if end_dt <= start_dt:
            raise ValueError("End time must be after start time.")

        # Create calendar and event
        event = Event()
        event.name = title
        event.begin = start_dt
        event.end = end_dt
        event.description = description or ""
        event.location = location or ""

        return event

    except ValueError as ve:
        print(f"Value error: {ve}")
