from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    APP_NAME: str = "CV Reader API"
    DEBUG: bool = True
    UPLOAD_DIR: Path = Path("uploads")

    # Supported file types
    ALLOWED_EXTENSIONS: list[str] = [".pdf", ".png", ".jpg", ".jpeg", ".webp", ".avif"]

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()

# Create upload directory if not exists
settings.UPLOAD_DIR.mkdir(exist_ok=True)
