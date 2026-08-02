import os
from pathlib import Path
from pydantic import BaseModel

class Settings(BaseModel):
    app_name: str = "Atlas CLI"
    version: str = "0.1.0"
    workspace_dir: Path = Path(".atlas_cli")
    
    db_name: str = os.getenv("DB_NAME", "atlas_db")
    db_host: str = os.getenv("DB_HOST", "postgres")
    db_port: str = os.getenv("DB_PORT", "5432")
    db_user: str = os.getenv("DB_USER", "atlas_user")
    db_password: str = os.getenv("DB_PASSWORD", "atlas_password")
    
    redis_host: str = os.getenv("REDIS_HOST", "redis")
    redis_port: str = os.getenv("REDIS_PORT", "6379")

    default_model: str = os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    nvidia_nim_api_key: str = os.getenv("NVIDIA_NIM_API_KEY", "")
    mistral_api_key: str = os.getenv("MISTRAL_API_KEY", "")
    together_api_key: str = os.getenv("TOGETHER_API_KEY", "")
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")

    @property
    def db_path(self) -> Path:
        return self.workspace_dir / f"{self.db_name}.db"

    @property
    def db_url(self) -> str:
        """Craft database URL dynamically before DB initiation."""
        custom_url = os.getenv("DATABASE_URL")
        if custom_url:
            return custom_url
        
        if self.db_host and self.db_host.lower() != "sqlite":
            return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
        
        return f"sqlite:///{self.db_path}"

    def ensure_workspace(self) -> None:
        """Ensure local workspace directory exists."""
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

settings = Settings()
