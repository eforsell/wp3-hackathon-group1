from fastmcp import FastMCP

from .mock_data import services

app = FastMCP(
    name="Service Catalog MCP",
)


@app.tool
def get_all_services():
    """Returns a list of all available services."""
    return services


@app.tool
def get_service_by_id(service_id: int):
    """Returns the service with the given ID."""
    for service in services:
        if service["id"] == service_id:
            return service
    return {"error": "Service not found."}


@app.tool
def get_service_by_name(name: str):
    """Returns the service with the given name."""
    for service in services:
        if service["name"].lower() == name.lower():
            return service
    return {"error": "Service not found."}


@app.tool
def get_services_by_category(category: str):
    """Returns all services in the given category."""
    matching_services = [
        service for service in services
        if service["category"].lower() == category.lower()]
    if matching_services:
        return matching_services
    return {"error": "No services found in this category."}


@app.tool
def get_service_access_requirements(service_name: str):
    """Returns the access requirements for a specific service."""
    for service in services:
        if service["name"].lower() == service_name.lower():
            return {
                "service_name": service["name"],
                "access_requirements": service["access_requirements"],
                "required_data": service["required_data"]
            }
    return {"error": "Service not found."}


@app.prompt
def purpose_prompt():
    return (
        "You are a service catalog assistant that helps users find available services "
        "and understand what data/requirements are needed to access them.")
