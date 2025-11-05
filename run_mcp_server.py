#!/usr/bin/env python3
"""Wrapper script to run the MCP server with correct PYTHONPATH.

This script sets the PYTHONPATH to the project root before running the server.
Use this in Cursor MCP configuration if the package is not installed.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Now import and run the server
from discovery.mcp.__main__ import main

if __name__ == "__main__":
    main()

