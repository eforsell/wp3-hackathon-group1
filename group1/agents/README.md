# Group 1 Agents

## Employee Catalog agent

First launch the MCP server the agent relies on:

```bash
uv run python -m  employee_catalog.employee_catalog_mcp
# localhost:8001
```

Launch the agent executor server (A2A) by running:

```bash
uv run python -m  employee_catalog
# localhost:8011
```
