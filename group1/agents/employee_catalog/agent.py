import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

load_dotenv()

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


if __name__ == "__main__":
    # Example of how to run the agent
    agent = await get_agent()
    response = agent.invoke({"input": "What is the name of the employee with id 1?"})
    print(response)
