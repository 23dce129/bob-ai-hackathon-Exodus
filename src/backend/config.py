"""
config.py — Application settings loaded from environment variables.

Every module that needs a config value imports from here.
Never call os.getenv() directly in other modules.
"""
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # Application
    app_port: int = Field(default=8000, alias="APP_PORT")
    app_env: str = Field(default="development", alias="APP_ENV")
    app_version: str = "0.1.0"

    # IBM watsonx.ai
    watsonx_api_key: str = Field(default="", alias="WATSONX_API_KEY")
    watsonx_project_id: str = Field(default="", alias="WATSONX_PROJECT_ID")
    watsonx_url: str = Field(
        default="https://us-south.ml.cloud.ibm.com", alias="WATSONX_URL"
    )

    # MCP server communication
    pharmaguard_api_url: str = Field(
        default="http://localhost:8000", alias="PHARMAGUARD_API_URL"
    )
    mcp_server_port: int = Field(default=8001, alias="MCP_SERVER_PORT")

    @property
    def watsonx_available(self) -> bool:
        """Returns True only when both required watsonx credentials are present."""
        return bool(self.watsonx_api_key and self.watsonx_project_id)

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


# Module-level singleton — import this everywhere
settings = Settings()
