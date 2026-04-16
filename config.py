from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).parent


class Settings(BaseSettings):
    # LLM
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b"
    ollama_temperature: float = 0.1
    ollama_num_ctx: int = 4096

    # Memory
    sqlite_db_path: str = str(BASE_DIR / "data" / "checkpoints.db")
    default_thread_id: str = "session-001"
    memory_window_size: int = 20

    # Observability
    log_level: str = "INFO"
    log_format: str = "text"  # "json" | "text"
    metrics_port: int = 8000
    enable_langsmith: bool = False
    langsmith_project: str = "purpo-culinary"

    # Web API
    api_host: str = "0.0.0.0"
    api_port: int = 8080

    # Evals
    evals_output_dir: str = str(BASE_DIR / "evals_results")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
