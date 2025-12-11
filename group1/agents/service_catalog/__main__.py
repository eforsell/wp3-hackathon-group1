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

from .agent import ServiceCatalogAgent
from .agent_executor import ServiceCatalogAgentExecutor

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MissingURLError(Exception):
    """Exception for missing URL."""


class MissingAPIKeyError(Exception):
    """Exception for missing API key."""


@click.command()
@click.option('--host', 'host', default='localhost')
@click.option('--port', 'port', default=8012)
def main(host, port):
    """Starts the Service Catalog Agent server."""
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
        skills = [
            AgentSkill(
                id='get_service_info',
                name='Get Service Information Tool',
                description=(
                    'Helps with retrieving information about services in the catalog'),
                tags=['service information'],
                examples=[
                    'What services exist in the Productivity category?',
                    'What service categories are there?',
                ],
            ),
            AgentSkill(
                id='get_service_access_requirements',
                name='Get Service Access Requirements',
                description=(
                    'Provides information about what is required for a user to be '
                    'given access to a particular service and what data is required to '
                    'request access.'),
                tags=['service access', 'service requirements'],
                examples=[
                    'What are the requirements of the user to access Office 365?',
                    'What is required to request access to the HR Information System?',
                ],
            ),
        ]
        agent_card = AgentCard(
            name='Service Catalog Agent',
            description=(
                'Helps with requests for access to different services'),
            url=f'http://{host}:{port}/',
            version='1.0.0',
            default_input_modes=ServiceCatalogAgent.SUPPORTED_CONTENT_TYPES,
            default_output_modes=ServiceCatalogAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=skills,
        )

        # --8<-- [start:DefaultRequestHandler]
        httpx_client = httpx.AsyncClient()
        push_config_store = InMemoryPushNotificationConfigStore()
        push_sender = BasePushNotificationSender(httpx_client=httpx_client,
                                                 config_store=push_config_store)
        request_handler = DefaultRequestHandler(
            agent_executor=ServiceCatalogAgentExecutor(),
            task_store=InMemoryTaskStore(),
            push_config_store=push_config_store,
            push_sender=push_sender
        )
        server = A2AStarletteApplication(
            agent_card=agent_card, http_handler=request_handler
        )

        uvicorn.run(server.build(), host=host, port=port)
        # --8<-- [end:DefaultRequestHandler]

    except MissingAPIKeyError as e:
        logger.error(f'Error: {e}')
        sys.exit(1)
    except Exception as e:
        logger.error(f'An error occurred during server startup: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()
    main()
