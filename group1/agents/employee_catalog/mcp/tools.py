from mcp.mock_data import employees


def get_all_employees():
    """Returns a list of all employees."""
    return employees


def get_employee_by_id(employee_id: int):
    """Returns the employee with the given ID."""
    for employee in employees:
        if employee["id"] == employee_id:
            return employee
    return {"error": "Employee not found."}


def get_employee_by_name(name: str):
    """Returns the employee with the given name."""
    for employee in employees:
        if employee["name"].lower() == name.lower():
            return employee
    return {"error": "Employee not found."}


tools = [get_all_employees, get_employee_by_id, get_employee_by_name]
