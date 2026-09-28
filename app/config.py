from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    APP_NAME: str = "CV Reader API"
    # When True, internal error messages are returned to the client.
    DEBUG: bool = True
    UPLOAD_DIR: Path = Path("uploads")
    POPPLER_PATH: Path | None = None

    # Supported file types
    ALLOWED_EXTENSIONS: list[str] = [".pdf", ".png", ".jpg", ".jpeg", ".webp", ".avif"]

    # Request limits
    MAX_UPLOAD_SIZE_MB: int = 20
    MAX_GROUND_TRUTH_CHARS: int = 50_000

    # Frontend origins allowed to call the API
    CORS_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ]

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()

# Create upload directory if not exists
settings.UPLOAD_DIR.mkdir(exist_ok=True)
