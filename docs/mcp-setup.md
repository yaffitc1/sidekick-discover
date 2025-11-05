# MCP Server Setup for Cursor

This guide explains how to add the Discovery Agent as an MCP server in Cursor.

## Prerequisites

1. Python 3.11+ installed
2. Discovery agent installed: `pip install -r requirements.txt`
3. Snowflake credentials configured via environment variables

## Installation

You have two options for installing the Discovery agent:

### Option 1: Install as a package (Recommended)

1. Install the package in development mode:
```bash
pip install -e .
```

This installs the package so Python can find the `discovery` module from anywhere.

2. Verify the MCP server can run:
```bash
python -m discovery.mcp
```

The server should start and wait for input via stdio (this is normal for MCP servers).

### Option 2: Use wrapper script (Quick setup)

If you prefer not to install the package, use the wrapper script instead.

## Cursor Configuration

### Option 1: Using installed package (Recommended)

If you installed the package with `pip install -e .`, use this configuration:

```json
{
  "mcpServers": {
    "discovery-agent": {
      "command": "python",
      "args": ["-m", "discovery.mcp"],
      "env": {
        "SNOWFLAKE_ACCOUNT": "your-account",
        "SNOWFLAKE_USER": "your-user",
        "SNOWFLAKE_PASSWORD": "your-password",
        "SNOWFLAKE_ROLE": "your-role",
        "SNOWFLAKE_WAREHOUSE": "your-warehouse"
      }
    }
  }
}
```

### Option 2: Using wrapper script (If package not installed)

If you didn't install the package, use the wrapper script with absolute path:

```json
{
  "mcpServers": {
    "discovery-agent": {
      "command": "python",
      "args": ["/absolute/path/to/sidekick-discover/run_mcp_server.py"],
      "env": {
        "SNOWFLAKE_ACCOUNT": "your-account",
        "SNOWFLAKE_USER": "your-user",
        "SNOWFLAKE_PASSWORD": "your-password",
        "SNOWFLAKE_ROLE": "your-role",
        "SNOWFLAKE_WAREHOUSE": "your-warehouse"
      }
    }
  }
}
```

**Note:** Replace `/absolute/path/to/sidekick-discover` with the actual absolute path to your project directory.

### Option 3: Using PYTHONPATH (Alternative)

You can also set PYTHONPATH in the environment:

```json
{
  "mcpServers": {
    "discovery-agent": {
      "command": "python",
      "args": ["-m", "discovery.mcp"],
      "env": {
        "PYTHONPATH": "/absolute/path/to/sidekick-discover",
        "SNOWFLAKE_ACCOUNT": "your-account",
        "SNOWFLAKE_USER": "your-user",
        "SNOWFLAKE_PASSWORD": "your-password",
        "SNOWFLAKE_ROLE": "your-role",
        "SNOWFLAKE_WAREHOUSE": "your-warehouse"
      }
    }
  }
}
```

Alternatively, if you're using a `.env` file in your project directory, you can omit the `env` section and the server will load variables from the `.env` file automatically.

### Environment Variables

The following environment variables are required for Snowflake connections:

- `SNOWFLAKE_ACCOUNT` - Your Snowflake account identifier
- `SNOWFLAKE_USER` - Your Snowflake username
- `SNOWFLAKE_PASSWORD` - Your Snowflake password
- `SNOWFLAKE_ROLE` - Your Snowflake role (optional, but recommended)
- `SNOWFLAKE_WAREHOUSE` - Your Snowflake warehouse name

You can also set these in a `.env` file in your project root:

```env
SNOWFLAKE_ACCOUNT=your-account
SNOWFLAKE_USER=your-user
SNOWFLAKE_PASSWORD=your-password
SNOWFLAKE_ROLE=your-role
SNOWFLAKE_WAREHOUSE=your-warehouse
```

## Available MCP Tools

Once configured, the following tools will be available in Cursor:

### 1. `register_snowflake_source`
Register a Snowflake source in the local registry.

**Parameters:**
- `name` (string, required): Alias name for the source
- `database` (string, required): Snowflake database name
- `schema` (string, required): Snowflake schema name
- `tables` (array of strings, optional): List of table names to include (if not provided, all tables are included)

**Example:**
```json
{
  "name": "sf_public",
  "database": "SNOWFLAKE_SAMPLE_DATA",
  "schema": "TPCH_SF1",
  "tables": ["ORDERS", "CUSTOMER"]
}
```

### 2. `run_discovery_agent`
Run the discovery agent pipeline on registered source(s) to analyze data, generate insights, and create dashboards.

**Parameters:**
- `source` (string, optional): Registered source name (for single source analysis)
- `sources` (string, optional): Comma-separated list of source names (for multi-source analysis)
- `sample_rows` (integer, optional): Maximum number of rows to sample per table (default: 100000)
- `goal` (string, optional): Optional exploration goal or focus area
- `output_dir` (string, optional): Base output directory path (default: "./outputs")

**Note:** Either `source` or `sources` must be provided, but not both.

**Example:**
```json
{
  "source": "sf_public",
  "sample_rows": 100000,
  "goal": "Find data quality issues",
  "output_dir": "./outputs"
}
```

### 3. `list_sources`
List all registered Snowflake sources in the local registry.

**Parameters:** None

**Returns:** JSON object with all registered sources and their configurations.

### 4. `get_dashboard_path`
Get the absolute path to the generated dashboard HTML file for a source.

**Parameters:**
- `source` (string, required): Registered source name
- `output_dir` (string, optional): Base output directory path (default: "./outputs")

**Example:**
```json
{
  "source": "sf_public",
  "output_dir": "./outputs"
}
```

## Usage Workflow

1. **Register a source:**
   - Use `register_snowflake_source` to register your Snowflake database/schema

2. **Run discovery:**
   - Use `run_discovery_agent` to analyze the registered source
   - The agent will generate profiles, insights, and a dashboard

3. **Access results:**
   - Use `get_dashboard_path` to get the path to the generated dashboard
   - Outputs are saved in `outputs/<source>/` directory

## Troubleshooting

### ModuleNotFoundError: No module named 'discovery'

This error occurs when Python can't find the `discovery` module. Fix it using one of these options:

**Option 1: Install the package (Recommended)**
```bash
pip install -e .
```

**Option 2: Use the wrapper script**
Update your MCP config to use the wrapper script with absolute path:
```json
{
  "mcpServers": {
    "discovery-agent": {
      "command": "python",
      "args": ["/absolute/path/to/sidekick-discover/run_mcp_server.py"]
    }
  }
}
```

**Option 3: Set PYTHONPATH**
Add PYTHONPATH to your MCP config env:
```json
{
  "mcpServers": {
    "discovery-agent": {
      "command": "python",
      "args": ["-m", "discovery.mcp"],
      "env": {
        "PYTHONPATH": "/absolute/path/to/sidekick-discover"
      }
    }
  }
}
```

To find your absolute path:
```bash
# On macOS/Linux
pwd

# Or in Python
python -c "from pathlib import Path; print(Path.cwd())"
```

### Server appears to do nothing when run directly
**This is normal!** MCP servers communicate via stdio and wait silently for JSON-RPC messages. When you run `python -m discovery.mcp` directly, it will:
- Start successfully
- Wait for input via stdin
- Appear to do nothing (this is expected)
- Respond when it receives JSON-RPC messages

To verify it's working:
```bash
python test_mcp_server.py
```

### Server won't start
- Verify Python is in your PATH: `python --version`
- Check that all dependencies are installed: `pip install -r requirements.txt`
- Ensure the command path is correct in your MCP configuration
- Run the test script: `python test_mcp_server.py` to verify functionality

### Connection errors
- Verify your Snowflake credentials are set correctly in environment variables
- Test credentials using the CLI: `discovery add snowflake --database ... --schema ... --name test`
- Check that your Snowflake account/warehouse is accessible

### Tool errors
- Ensure sources are registered before running the agent
- Check that output directories are writable
- Review error messages returned by tools for specific issues

## Security Notes

- Never commit credentials to version control
- Use environment variables or `.env` files (which should be in `.gitignore`)
- Use least-privilege Snowflake roles when possible
- The MCP server runs locally and connects to Snowflake remotely; no data is stored in the server itself

