from fastmcp import FastMCP
from mcp.types import PromptMessage, TextContent

from .mock_data import employees
from .prompts import BASE_INSTRUCTIONS

app = FastMCP(
    name="Employee Catalog MCP",
)


@app.tool
def get_all_employees():
    """Returns a list of all employees."""
    return employees


@app.tool
def get_employee_by_id(employee_id: int):
    """Returns the employee with the given ID."""
    for employee in employees:
        if employee["id"] == employee_id:
            return employee
    return {"error": "Employee not found."}


@app.tool
def get_employee_by_name(name: str):
    """Returns the employee with the given name."""
    for employee in employees:
        if employee["name"].lower() == name.lower():
            return employee
    return {"error": "Employee not found."}


@app.prompt
def base_prompt():
    return PromptMessage(
        role="assistant", content=TextContent(type="text", text=BASE_INSTRUCTIONS))


if __name__ == "__main__":
    app.run(transport="http", host="0.0.0.0", port=8002)
