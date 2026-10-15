"""Launcher for the MCP server that works from any directory: `python scripts/run_mcp_server.py`."""

from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mcp_server.__main__ import main

if __name__ == "__main__":
    main()
