import asyncio
import os

import langchain.messages
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

load_dotenv(override=True)

client = MultiServerMCPClient({  # type: ignore
    "service_catalog": {
        "transport": "http",
        "url": "http://0.0.0.0:8003/mcp"
    }
})

llm = ChatOpenAI(
    model="gpt-5.1",
    reasoning_effort='minimal',
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
    base_messages = await client.get_prompt("service_catalog", "purpose_prompt")
    messages = [
        langchain.messages.SystemMessage("""\
            Du är en hjälpsam assistent som har till uppgift att hämta och tillhandahålla 
            information från Tjänstekatalogen. Var vänlig och hjälpsam, och ge tydlig 
            information om tillgängliga tjänster och deras krav.
            """),
        * base_messages,
        langchain.messages.HumanMessage("What services are available in the 'Productivity' category?")
    ]

    async for event in agent.astream_events({'messages': messages}):
        print(event)


if __name__ == "__main__":
    asyncio.run(main())
