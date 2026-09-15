"""
config.py — MCP server configuration.

Loaded once at startup. All values come from environment variables so the
server can be pointed at different backend instances without code changes.

Usage
-----
    from config import mcp_settings

    base = mcp_settings.api_base_url          # "http://localhost:8000"
    timeout = mcp_settings.request_timeout    # 30.0 (seconds)
"""
from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class MCPSettings(BaseSettings):
    """Environment-driven configuration for the PharmaGuard MCP server."""

    # URL of the running PharmaGuard FastAPI backend.
    # Change if the backend is on a different host or port.
    api_base_url: str = Field(
        default="http://localhost:8000",
        alias="PHARMAGUARD_API_URL",
        description="Base URL of the PharmaGuard FastAPI backend.",
    )

    # HTTP request timeout for backend calls (seconds).
    # Increase if the backend is slow to start or is running on a remote host.
    request_timeout: float = Field(
        default=30.0,
        alias="PHARMAGUARD_MCP_TIMEOUT",
        description="HTTP timeout in seconds for backend API calls.",
    )

    # Maximum number of signals/entries to return in tool responses.
    # Keeps MCP tool responses concise enough for Bob to process.
    max_results: int = Field(
        default=10,
        alias="PHARMAGUARD_MCP_MAX_RESULTS",
        description="Maximum signals / sections to include in tool response.",
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "populate_by_name": True,
    }


# Module-level singleton
mcp_settings = MCPSettings()
