"""Entry point for running the MCP server.

Run with: python -m discovery.mcp
"""

import sys
import asyncio
from discovery.mcp.server import run_server


def main() -> None:
    """Run the MCP server."""
    # MCP servers communicate via stdio, so they appear silent when running
    # This is normal - they wait for JSON-RPC messages from the MCP client
    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        # Graceful shutdown on Ctrl+C
        sys.exit(0)
    except Exception as e:
        # Print errors to stderr so they're visible
        print(f"Error starting MCP server: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

