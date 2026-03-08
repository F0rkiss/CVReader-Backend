from fastapi import UploadFile
from pathlib import Path
from app.config import settings
import uuid
import os


async def save_upload_file(upload_file: UploadFile) -> str:
    """
    Save an uploaded file to the uploads directory.
    Returns the file path.
    """
    # Generate unique filename
    file_ext = os.path.splitext(upload_file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    file_path = settings.UPLOAD_DIR / unique_filename
    
    # Save file
    content = await upload_file.read()
    with open(file_path, "wb") as f:
        f.write(content)
    
    return str(file_path)


def cleanup_file(file_path: str) -> None:
    """Remove a file from the filesystem."""
    try:
        Path(file_path).unlink(missing_ok=True)
    except Exception:
        pass  # Ignore cleanup errors
