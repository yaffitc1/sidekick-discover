"""MCP server implementation for Discovery agent.

Exposes discovery agent functionality as MCP tools.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from discovery.io.registry import upsert_source, get_source, load_registry
from discovery.agent.orchestrator import run_snowflake, run_multi_source

# Load environment variables from .env file
load_dotenv()

# Create MCP server instance
app = Server("discovery-agent")


@app.list_tools()
async def handle_list_tools() -> List[Tool]:
    """List available MCP tools."""
    return [
        Tool(
            name="register_snowflake_source",
            description="Register a Snowflake source in the local registry. Credentials must be set via environment variables (SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD, SNOWFLAKE_ROLE, SNOWFLAKE_WAREHOUSE).",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Alias name for the source"
                    },
                    "database": {
                        "type": "string",
                        "description": "Snowflake database name"
                    },
                    "schema": {
                        "type": "string",
                        "description": "Snowflake schema name"
                    },
                    "tables": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of table names to include (if not provided, all tables are included)"
                    }
                },
                "required": ["name", "database", "schema"]
            }
        ),
        Tool(
            name="run_discovery_agent",
            description="Run the discovery agent pipeline on registered Snowflake source(s) to analyze data, generate insights, and create dashboards.",
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Registered source name (for single source analysis)"
                    },
                    "sources": {
                        "type": "string",
                        "description": "Comma-separated list of source names (for multi-source analysis). Cannot be used with 'source'."
                    },
                    "sample_rows": {
                        "type": "integer",
                        "description": "Maximum number of rows to sample per table",
                        "default": 100000
                    },
                    "goal": {
                        "type": "string",
                        "description": "Optional exploration goal or focus area"
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Base output directory path",
                        "default": "./outputs"
                    }
                }
            }
        ),
        Tool(
            name="list_sources",
            description="List all registered Snowflake sources in the local registry.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="get_dashboard_path",
            description="Get the absolute path to the generated dashboard HTML file for a source.",
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Registered source name"
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Base output directory path",
                        "default": "./outputs"
                    }
                },
                "required": ["source"]
            }
        )
    ]


@app.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any] | None) -> List[TextContent]:
    """Handle tool calls."""
    if arguments is None:
        arguments = {}
    
    if name == "register_snowflake_source":
        name_param = arguments.get("name")
        database = arguments.get("database")
        schema = arguments.get("schema")
        tables = arguments.get("tables")
        
        if not name_param or not database or not schema:
            return [TextContent(
                type="text",
                text=json.dumps({"error": "Missing required parameters: name, database, schema"})
            )]
        
        src = {
            "type": "snowflake",
            "database": database,
            "schema": schema,
            "tables": tables if tables else None,
        }
        
        try:
            upsert_source(name_param, src)
            tables_str = f" tables: {', '.join(tables)}" if tables else " (all tables)"
            result = {
                "success": True,
                "message": f"Registered Snowflake source '{name_param}' -> {database}.{schema}{tables_str}",
                "source": {
                    "name": name_param,
                    "database": database,
                    "schema": schema,
                    "tables": tables
                }
            }
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"error": f"Failed to register source: {str(e)}"})
            )]
    
    elif name == "run_discovery_agent":
        source = arguments.get("source")
        sources = arguments.get("sources")
        sample_rows = arguments.get("sample_rows", 100000)
        goal = arguments.get("goal")
        output_dir = arguments.get("output_dir", "./outputs")
        
        if not source and not sources:
            return [TextContent(
                type="text",
                text=json.dumps({"error": "Either 'source' or 'sources' must be provided"})
            )]
        
        if source and sources:
            return [TextContent(
                type="text",
                text=json.dumps({"error": "Cannot use both 'source' and 'sources' parameters"})
            )]
        
        try:
            # Multi-source mode
            if sources:
                source_list = [s.strip() for s in sources.split(",")]
                source_configs = []
                
                for src_name in source_list:
                    src = get_source(src_name)
                    if not src:
                        return [TextContent(
                            type="text",
                            text=json.dumps({"error": f"Source '{src_name}' not found. Register it first using register_snowflake_source."})
                        )]
                    
                    if src.get("type") != "snowflake":
                        return [TextContent(
                            type="text",
                            text=json.dumps({"error": f"Only Snowflake sources are supported. Source '{src_name}' has type '{src.get('type')}'"})
                        )]
                    
                    source_configs.append({
                        "name": src_name,
                        "database": src["database"],
                        "schema": src["schema"],
                        "tables": src.get("tables"),
                    })
                
                output_path = Path(output_dir) / "multi_source_analysis"
                run_multi_source(
                    sources=source_configs,
                    output_dir=str(output_path),
                    sample_rows=sample_rows,
                    goal=goal,
                )
                
                result = {
                    "success": True,
                    "message": "Multi-source analysis completed",
                    "output_dir": str(output_path.resolve()),
                    "dashboard_path": str((output_path / "dashboard.html").resolve()),
                    "sources": source_list
                }
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
            # Single source mode
            src = get_source(source)
            if not src:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": f"Source '{source}' not found. Register it first using register_snowflake_source."})
                )]
            
            if src.get("type") != "snowflake":
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": f"Only Snowflake sources are supported. Source '{source}' has type '{src.get('type')}'"})
                )]
            
            output_path = Path(output_dir) / source
            run_snowflake(
                database=src["database"],
                schema=src["schema"],
                source_name=source,
                output_dir=str(output_path),
                tables=src.get("tables"),
                sample_rows=sample_rows,
                goal=goal,
            )
            
            dashboard_path = output_path / "dashboard.html"
            result = {
                "success": True,
                "message": "Agent run completed",
                "output_dir": str(output_path.resolve()),
                "dashboard_path": str(dashboard_path.resolve()),
                "source": source
            }
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"error": f"Failed to run discovery agent: {str(e)}"})
            )]
    
    elif name == "list_sources":
        try:
            registry = load_registry()
            result = {
                "success": True,
                "sources": registry
            }
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"error": f"Failed to list sources: {str(e)}"})
            )]
    
    elif name == "get_dashboard_path":
        source = arguments.get("source")
        output_dir = arguments.get("output_dir", "./outputs")
        
        if not source:
            return [TextContent(
                type="text",
                text=json.dumps({"error": "Missing required parameter: source"})
            )]
        
        try:
            dashboard_path = Path(output_dir) / source / "dashboard.html"
            resolved_path = dashboard_path.resolve()
            
            if not resolved_path.exists():
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": f"Dashboard not found for source '{source}'. Run discovery agent first.",
                        "expected_path": str(resolved_path)
                    })
                )]
            
            result = {
                "success": True,
                "dashboard_path": str(resolved_path),
                "source": source
            }
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"error": f"Failed to get dashboard path: {str(e)}"})
            )]
    
    else:
        return [TextContent(
            type="text",
            text=json.dumps({"error": f"Unknown tool: {name}"})
        )]


def create_server() -> Server:
    """Create and return the MCP server instance."""
    return app


async def run_server():
    """Run the MCP server using stdio transport."""
    from mcp.server.stdio import stdio_server
    
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )

