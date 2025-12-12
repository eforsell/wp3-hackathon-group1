import logging
import os
import sys

import click
import httpx
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import (BasePushNotificationSender,
                              InMemoryPushNotificationConfigStore, InMemoryTaskStore)
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from dotenv import load_dotenv

from .agent import CoordinatorAgent
from .agent_executor import CoordinatorAgentExecutor

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MissingURLError(Exception):
    """Exception for missing URL."""


class MissingAPIKeyError(Exception):
    """Exception for missing API key."""


@click.command()
@click.option('--host', 'host', default='localhost')
@click.option('--port', 'port', default=10003)
def main(host, port):
    """Starts the Coordinator Agent server."""
    try:
        if not os.getenv('OPENAI_BASE_URL'):
            raise MissingURLError(
                'OPENAI_BASE_URL environment variable not set.'
            )
        if not os.getenv('OPENAI_API_KEY'):
            raise MissingAPIKeyError(
                'OPENAI_API_KEY environment variable not set.'
            )

        capabilities = AgentCapabilities(streaming=True, push_notifications=True)
        skill = AgentSkill(
            id='coordinate_agents',
            name='Multi-Agent Coordination Tool',
            description='Coordinates queries across employee and service catalog agents',
            tags=['coordination', 'employee catalog', 'service catalog', 'delegation'],
            examples=[
                'Find employees and their assigned services',
                'List all services and who uses them',
                'Which employees have access to AWS Development Account?',
            ],
        )
        agent_card = AgentCard(
            name='Coordinator Agent',
            description='Intelligently routes queries to employee and service catalog agents',
            url=f'http://{host}:{port}/',
            version='1.0.0',
            default_input_modes=CoordinatorAgent.SUPPORTED_CONTENT_TYPES,
            default_output_modes=CoordinatorAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill],
        )

        httpx_client = httpx.AsyncClient()
        push_config_store = InMemoryPushNotificationConfigStore()
        push_sender = BasePushNotificationSender(httpx_client=httpx_client,
                                                 config_store=push_config_store)
        request_handler = DefaultRequestHandler(
            agent_executor=CoordinatorAgentExecutor(),
            task_store=InMemoryTaskStore(),
            push_config_store=push_config_store,
            push_sender=push_sender
        )
        server = A2AStarletteApplication(
            agent_card=agent_card, http_handler=request_handler
        )

        uvicorn.run(server.build(), host=host, port=port)

    except MissingAPIKeyError as e:
        logger.error(f'Error: {e}')
        sys.exit(1)
    except Exception as e:
        logger.error(f'An error occurred during server startup: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()
