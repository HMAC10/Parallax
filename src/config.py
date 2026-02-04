"""
Configuration module using Pydantic Settings.

Loads environment variables from .env file and provides
type-safe access to all configuration values.
"""

from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Create a .env file based on .env.example and fill in your values.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # ==========================================================================
    # NVIDIA API Configuration
    # ==========================================================================
    
    nvidia_api_key: str = Field(
        default="",
        description="NVIDIA API key from build.nvidia.com"
    )
    
    nvidia_base_url: str = Field(
        default="https://integrate.api.nvidia.com/v1",
        description="NVIDIA NIM API base URL"
    )
    
    # ==========================================================================
    # Model Configuration
    # ==========================================================================
    
    nemotron_model: str = Field(
        default="nvidia/llama-3.1-nemotron-70b-instruct",
        description="Nemotron model for command parsing"
    )
    
    vila_model: str = Field(
        default="nvidia/vila",
        description="VILA model for image analysis"
    )
    
    # ==========================================================================
    # Drone Configuration
    # ==========================================================================
    
    use_mock_drone: bool = Field(
        default=True,
        description="Use mock drone for testing"
    )
    
    isaac_sim_host: str = Field(
        default="localhost",
        description="Isaac Sim host address"
    )
    
    isaac_sim_port: int = Field(
        default=8211,
        description="Isaac Sim port"
    )
    
    # ==========================================================================
    # Logging Configuration
    # ==========================================================================
    
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    
    verbose_api: bool = Field(
        default=False,
        description="Enable verbose API logging"
    )
    
    @property
    def is_configured(self) -> bool:
        """Check if the essential API keys are configured."""
        return bool(self.nvidia_api_key and not self.nvidia_api_key.startswith("nvapi-xxx"))
    
    def get_headers(self) -> dict:
        """Get authorization headers for NVIDIA API calls."""
        return {
            "Authorization": f"Bearer {self.nvidia_api_key}",
            "Content-Type": "application/json",
        }


# Global settings instance
settings = Settings()


def validate_config() -> tuple[bool, list[str]]:
    """
    Validate the current configuration.
    
    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []
    
    if not settings.nvidia_api_key:
        errors.append("NVIDIA_API_KEY is not set")
    elif settings.nvidia_api_key.startswith("nvapi-xxx"):
        errors.append("NVIDIA_API_KEY appears to be a placeholder value")
    
    if not settings.nvidia_base_url:
        errors.append("NVIDIA_BASE_URL is not set")
    
    return len(errors) == 0, errors


if __name__ == "__main__":
    # Quick config test
    from rich import print as rprint
    from rich.panel import Panel
    
    rprint(Panel.fit(
        f"[bold]Configuration Status[/bold]\n\n"
        f"API Key Set: {bool(settings.nvidia_api_key)}\n"
        f"Base URL: {settings.nvidia_base_url}\n"
        f"Nemotron Model: {settings.nemotron_model}\n"
        f"VILA Model: {settings.vila_model}\n"
        f"Mock Drone: {settings.use_mock_drone}\n"
        f"Log Level: {settings.log_level}",
        title="Drone AI Config"
    ))
    
    is_valid, errors = validate_config()
    if not is_valid:
        rprint("[red]Configuration errors:[/red]")
        for err in errors:
            rprint(f"  • {err}")
