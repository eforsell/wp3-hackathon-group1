# Gemini Code Assistant Context

This document provides context for the Gemini code assistant to understand the project and assist in its development.

## Project Overview

This is a Python project to build a set of agents for an onboarding system. The goal is to automate and streamline the onboarding process for new employees.

The project uses `langchain` and `langgraph` to build these agents. The main entry point is `main.py`.

The project structure includes an `agents` directory, which will contain the different agents. Currently, two agents are planned:

* **Service Catalog Agent:** To handle employee tasks and service requests.
* **Employee Catalog Agent:** To manage employee data.

## Building and Running

* **Install dependencies:**

    ```bash
    uv pip install -e .
    uv add fastapi uvicorn langserve
    ```

* **Run the application:**

    ```bash
    python main.py
    ```

* **Run tests:**
    There are no tests yet, but the `pyproject.toml` is configured to use `pytest`. Tests should be placed in a `tests` directory.

## Development Conventions

* **Linting and Formatting:** The project is configured to use `flake8` for linting and `autopep8` for formatting. The max line length is 88 characters.
* **Typing:** `pyright` is configured for basic type checking.
* **Dependencies:** Project dependencies are managed in `pyproject.toml`.

## Quality Assurance

After making any code changes, it is important to run the following quality checks. These tasks are run using `uv run`.

* **Linting:**

    ```bash
    uv run flake8 .
    ```

* **Type Checking:**

    ```bash
    uv run pyright .
    ```

### Error Handling

* **Simple Errors:** For simple `flake8` and `pyright` errors, the agent should notify the user and let them fix the issues.
* **Complex Errors:** For more complex or fundamental errors (e.g., complex type errors, architectural issues), the agent should attempt to fix them.

## File Index

* `README.md`: Contains a high-level overview of the project and setup instructions.
* `pyproject.toml`: Defines project metadata, dependencies, and tool configurations.
* `main.py`: The main entry point of the application.
* `agents/`: Directory containing the agent implementations.
  * `service_catalog/`: The service catalog agent.
  * `employee_catalog/`: The employee catalog agent.

## Agent Specific Guidelines

* **Langchain/Langgraph Documentation:** When making changes to `langchain` or `langgraph` code, always consult the `langchain-docs MCP server` for the most up-to-date documentation.
* **FastMCP Documentation:** When making changes to `fastmcp` code, always consult the `fastmcp-docs MCP server` for the most up-to-date documentation.
