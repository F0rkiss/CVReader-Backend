from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    APP_NAME: str = "CV Reader API"
    DEBUG: bool = True
    UPLOAD_DIR: Path = Path("uploads")
    
    # Supported file types
    ALLOWED_EXTENSIONS: list = [".pdf", ".png", ".jpg", ".jpeg"]
    
    class Config:
        env_file = ".env"


settings = Settings()

# Create upload directory if not exists
settings.UPLOAD_DIR.mkdir(exist_ok=True)
