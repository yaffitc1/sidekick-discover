"""MCP server for Discovery agent.

Exposes discovery agent functionality as MCP tools for use in Cursor.
"""

from discovery.mcp.server import create_server

__all__ = ["create_server"]

