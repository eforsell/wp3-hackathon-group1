import asyncio
import os

import langchain.messages
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

load_dotenv(override=True)

client = MultiServerMCPClient({  # type: ignore
    "employee_catalog": {
        "transport": "http",
        "url": "http://0.0.0.0:8002/mcp"
    }
})

llm = ChatOpenAI(
    model="gpt-4o-mini",
    base_url=os.environ["OPENAI_BASE_URL"],
    api_key=lambda: os.environ["OPENAI_API_KEY"],
)


async def get_agent():
    # Create the agent

    tools = await client.get_tools()
    return create_agent(
        model=llm,
        tools=tools
    )


async def main():
    # Example of how to run the agent
    agent = await get_agent()
    messages = [
        {'role': 'system', 'content': "You are a helpful assistant"},
        {'role': 'user', 'content': "What is the name of the employee with id 1?"},
    ]
    response = await agent.ainvoke({'messages': messages})
    print(response)
    # messages.append(response)

    # # Print messages
    # for message in messages:
    #     print()


if __name__ == "__main__":
    asyncio.run(main())
