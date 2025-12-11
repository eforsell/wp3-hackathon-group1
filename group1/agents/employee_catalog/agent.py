import requests
import yaml
from langchain_community.agent_toolkits.openapi.planner import create_openapi_agent
from langchain_community.agent_toolkits.openapi.spec import ReducedOpenAPISpec
from langchain_community.utilities.requests import TextRequestsWrapper
from langchain_openai import ChatOpenAI

# Load the OpenAPI spec from the running MCP server
response = requests.get("http://localhost:8002/openapi.json")
spec_dict = yaml.safe_load(response.text)
spec = ReducedOpenAPISpec.from_dict(spec_dict)

# Create the agent
llm = ChatOpenAI(model="gpt-3.5-turbo")
requests_wrapper = TextRequestsWrapper()
agent = create_openapi_agent(
    spec,
    requests_wrapper,
    llm,
    allow_dangerous_requests=True,
)


if __name__ == "__main__":
    # Example of how to run the agent
    response = agent.invoke({"input": "What is the name of the employee with id 1?"})
    print(response)
