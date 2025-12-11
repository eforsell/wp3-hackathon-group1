import asyncio
import os
from collections.abc import AsyncIterable
from typing import Any, Literal

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
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


class ServiceCatalogAgent:
    """ServiceCatalogAgent - a specialized assistant for service catalog queries."""

    SYSTEM_INSTRUCTION = (
        'You are a helpful assistant tasked with retrieving and providing information '
        'from the Service Catalog. Be friendly and helpful, and provide clear '
        'information about available services and their requirements.'
    )

    def __init__(self):
        self.model = ChatOpenAI(
            model="gpt-5.1",
            reasoning_effort='minimal',
            base_url=os.environ["OPENAI_BASE_URL"],
            api_key=lambda: os.environ["OPENAI_API_KEY"],
        )

        self.mcp_client = MultiServerMCPClient({  # type: ignore
            "service_catalog": {
                "transport": "http",
                "url": "http://0.0.0.0:8002/mcp"
            }
        })
        self.tools = asyncio.run(self.mcp_client.get_tools())

        self.graph = create_agent(
            model=self.model,
            tools=self.tools,
            checkpointer=memory,
            system_prompt=self.SYSTEM_INSTRUCTION,
            response_format=ResponseFormat,
        )

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
                yield {
                    'is_task_complete': False,
                    'require_user_input': False,
                    'content': 'Interacting with the service catalog...',
                }
            elif isinstance(message, ToolMessage):
                yield {
                    'is_task_complete': False,
                    'require_user_input': False,
                    'content': 'Processing the results...',
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
