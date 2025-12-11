import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

load_dotenv()

client = MultiServerMCPClient({  # type: ignore
    "service_catalog": {
        "transport": "http",
        "url": "http://0.0.0.0:8003/mcp"
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


if __name__ == "__main__":
    # Example of how to run the agent
    agent = await get_agent()
    response = agent.invoke({"input": "What services are available in the 'data' category?"})
    print(response)
