import asyncio
import os
from collections.abc import AsyncIterable
from typing import Any, Literal

import httpx
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

memory = MemorySaver()

load_dotenv(override=True)


class ResponseFormat(BaseModel):
    """Respond to the user in this format."""

    status: Literal['input_required', 'completed', 'error'] = Field(
        default='input_required',
        description=(
            'Set response status to input_required if the user needs to provide more '
            'information to complete the request. Set response status to error if '
            'there is an error while processing the request. Set response status to '
            'completed if the request is complete.'
        )
    )
    message: str


class A2AAgentTool(BaseModel):
    """Tool that delegates to another agent via A2A framework."""

    name: str
    agent_url: str
    description: str

    async def invoke(self, query: str, context_id: str) -> str:
        """Invoke another agent via A2A framework."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.agent_url}/tasks",
                json={"message": query, "context_id": context_id}
            )
            return response.json().get('result', 'No response from agent')


class CoordinatorAgent:
    """CoordinatorAgent - orchestrates employee and service catalog agents via A2A framework."""

    SYSTEM_INSTRUCTION = (
        'Du är en hjälpsam och intelligent koordinator som har till uppgift att '
        'dirigera användarförfrågningar till rätt agent. Du kan delegera uppgifter '
        'till EmployeeCatalogAgent för personalkatalogfrågor och ServiceCatalogAgent '
        'för tjänstekatalogfrågor. Analysera användarens fråga och bestäm vilken '
        'agent som bäst kan hantera förfrågan. Ha en vänlig och hjälpsam ton.'
    )

    # Agent URLs (configured via environment variables or defaults)
    AGENT_URLS = {
        'employee_catalog': os.environ.get('EMPLOYEE_CATALOG_AGENT_URL', 'http://localhost:10001'),
        'service_catalog': os.environ.get('SERVICE_CATALOG_AGENT_URL', 'http://localhost:10002'),
    }

    def __init__(self):
        self.model = ChatOpenAI(
            model="gpt-5.1",
            reasoning_effort='minimal',
            base_url=os.environ["OPENAI_BASE_URL"],
            api_key=lambda: os.environ["OPENAI_API_KEY"],
        )

        # Create A2A agent tools
        self.agent_tools = {
            'employee_catalog_tool': A2AAgentTool(
                name='query_employee_catalog',
                agent_url=self.AGENT_URLS['employee_catalog'],
                description='Query the Employee Catalog Agent for information about employees'
            ),
            'service_catalog_tool': A2AAgentTool(
                name='query_service_catalog',
                agent_url=self.AGENT_URLS['service_catalog'],
                description='Query the Service Catalog Agent for information about services'
            ),
        }

        self.graph = create_agent(
            model=self.model,
            tools=[self._create_tool(name, tool) for name, tool in self.agent_tools.items()],
            checkpointer=memory,
            system_prompt=self.SYSTEM_INSTRUCTION,
            response_format=ResponseFormat,
        )

    def _create_tool(self, tool_id: str, agent_tool: A2AAgentTool):
        """Create a LangChain tool from an A2A agent tool."""
        async def tool_func(query: str, context_id: str = "") -> str:
            return await agent_tool.invoke(query, context_id)

        tool_func.__doc__ = agent_tool.description
        tool_func.__name__ = agent_tool.name
        return tool_func

    async def stream(self, query, context_id) -> AsyncIterable[dict[str, Any]]:
        inputs = {'messages': [('user', query)]}
        config = {'configurable': {'thread_id': context_id}}

        for item in self.graph.stream(
                input=inputs, config=config, stream_mode='values'):  # type: ignore
            message = item['messages'][-1]
            if (
                isinstance(message, AIMessage)
                and message.tool_calls
                and len(message.tool_calls) > 0
            ):
                tool_names = [tc.get('name', 'unknown') for tc in message.tool_calls]
                yield {
                    'is_task_complete': False,
                    'require_user_input': False,
                    'content': f'Delegating to: {", ".join(tool_names)}...',
                }
            elif isinstance(message, ToolMessage):
                yield {
                    'is_task_complete': False,
                    'require_user_input': False,
                    'content': 'Processing the results from agents...',
                }

        yield self.get_agent_response(config)

    def get_agent_response(self, config):
        current_state = self.graph.get_state(config)
        structured_response = current_state.values.get('structured_response')
        if structured_response and isinstance(
            structured_response, ResponseFormat
        ):
            if structured_response.status == 'input_required':
                return {
                    'is_task_complete': False,
                    'require_user_input': True,
                    'content': structured_response.message,
                }
            if structured_response.status == 'error':
                return {
                    'is_task_complete': False,
                    'require_user_input': True,
                    'content': structured_response.message,
                }
            if structured_response.status == 'completed':
                return {
                    'is_task_complete': True,
                    'require_user_input': False,
                    'content': structured_response.message,
                }

        return {
            'is_task_complete': False,
            'require_user_input': True,
            'content': (
                'We are unable to process your request at the moment. '
                'Please try again.'
            ),
        }

    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']
