from fastmcp import FastMCP
from mcp.tools import tools

app = FastMCP(
    name="Employee Catalog MCP",
    tools=tools,
)

if __name__ == "__main__":
    app.run(transport="http", host="0.0.0.0", port=8002)
