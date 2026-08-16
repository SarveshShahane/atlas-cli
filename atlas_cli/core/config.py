import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel

# Try loading from cwd, then fallback to project root
load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)
_root_env = Path(__file__).resolve().parent.parent.parent / ".env"
if _root_env.exists():
    load_dotenv(dotenv_path=_root_env, override=False)


class Settings(BaseModel):
    app_name: str = "Atlas CLI"
    version: str = "0.1.0"

    @property
    def workspace_dir(self) -> Path:
        env_dir = os.getenv("ATLAS_WORKSPACE_DIR")
        if env_dir:
            return Path(env_dir)
        local_dir = Path(".atlas_cli")
        if local_dir.exists() and (local_dir / "runs").exists():
            return local_dir
        project_root = Path(__file__).resolve().parent.parent.parent / ".atlas_cli"
        if project_root.exists():
            return project_root
        return local_dir

    @property
    def db_name(self) -> str:
        return os.getenv("DB_NAME", "atlas_db")

    @property
    def db_host(self) -> str:
        return os.getenv("DB_HOST", "postgres")

    @property
    def db_port(self) -> str:
        return os.getenv("DB_PORT", "5432")

    @property
    def db_user(self) -> str:
        return os.getenv("DB_USER", "atlas_user")

    @property
    def db_password(self) -> str:
        return os.getenv("DB_PASSWORD", "atlas_password")

    @property
    def redis_host(self) -> str:
        return os.getenv("REDIS_HOST", "redis")

    @property
    def redis_port(self) -> str:
        return os.getenv("REDIS_PORT", "6379")

    @property
    def llm_model(self) -> str:
        return os.getenv("LLM_MODEL", os.getenv("DEFAULT_LLM_MODEL", "groq/llama-3.3-70b-versatile"))

    @property
    def openai_api_key(self) -> str:
        return os.getenv("OPENAI_API_KEY", "")

    @property
    def anthropic_api_key(self) -> str:
        return os.getenv("ANTHROPIC_API_KEY", "")

    @property
    def gemini_api_key(self) -> str:
        return os.getenv("GEMINI_API_KEY", "")

    @property
    def groq_api_key(self) -> str:
        return os.getenv("GROQ_API_KEY", "")

    @property
    def nvidia_nim_api_key(self) -> str:
        return os.getenv("NVIDIA_NIM_API_KEY", "")

    @property
    def mistral_api_key(self) -> str:
        return os.getenv("MISTRAL_API_KEY", "")

    @property
    def together_api_key(self) -> str:
        return os.getenv("TOGETHER_API_KEY", "")

    @property
    def openrouter_api_key(self) -> str:
        return os.getenv("OPENROUTER_API_KEY", "")

    @property
    def default_model(self) -> str:
        """Alias for backwards compatibility."""
        return self.llm_model

    @property
    def db_path(self) -> Path:
        return self.workspace_dir / f"{self.db_name}.db"

    @property
    def db_url(self) -> str:
        """Craft database URL dynamically before DB initiation."""
        custom_url = os.getenv("DATABASE_URL", "").strip()
        if custom_url:
            return custom_url
        return f"sqlite:///{self.db_path}"

    def ensure_workspace(self) -> None:
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        (self.workspace_dir / "runs").mkdir(parents=True, exist_ok=True)
        (self.workspace_dir / "logs").mkdir(parents=True, exist_ok=True)
        (self.workspace_dir / "models").mkdir(parents=True, exist_ok=True)


settings = Settings()
