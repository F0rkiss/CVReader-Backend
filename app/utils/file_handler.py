from fastapi import HTTPException, UploadFile
from pathlib import Path
from app.config import settings
import logging
import uuid
import os

logger = logging.getLogger(__name__)


async def save_upload_file(upload_file: UploadFile) -> str:
    """
    Save an uploaded file to the uploads directory.
    Returns the file path.

    Raises HTTPException(413) when the file exceeds MAX_UPLOAD_SIZE_MB.
    """
    # Generate unique filename
    file_ext = os.path.splitext(upload_file.filename or "")[1].lower()
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    file_path = settings.UPLOAD_DIR / unique_filename

    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    written = 0
    too_large = False

    # Save file (streaming to avoid loading large PDFs into memory)
    chunk_size = 1024 * 1024  # 1MB
    with open(file_path, "wb") as f:
        while True:
            chunk = await upload_file.read(chunk_size)
            if not chunk:
                break
            written += len(chunk)
            if written > max_bytes:
                too_large = True
                break
            f.write(chunk)

    if too_large:
        cleanup_file(str(file_path))
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {settings.MAX_UPLOAD_SIZE_MB} MB.",
        )

    return str(file_path)


def cleanup_file(file_path: str) -> None:
    """Remove a file from the filesystem."""
    try:
        Path(file_path).unlink(missing_ok=True)
    except OSError:
        logger.warning("Failed to remove temporary upload %s", file_path, exc_info=True)
