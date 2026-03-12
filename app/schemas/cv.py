from pydantic import BaseModel
from typing import Literal, Optional


class ClassificationResponse(BaseModel):
    """Response model for CV classification"""
    filename: str
    cv_type: Literal["ATS", "Creative"]
    confidence: float
    details: dict

    class Config:
        json_schema_extra = {
            "example": {
                "filename": "resume.pdf",
                "cv_type": "ATS",
                "confidence": 0.85,
                "details": {
                    "complexity_score": 0.2,
                    "is_colorful": False,
                    "has_multiple_columns": False,
                },
            }
        }


class OCRResponse(BaseModel):
    """Response model for full CV reading"""
    filename: str
    cv_type: Literal["ATS", "Creative"]
    classification_confidence: float
    ocr_engine: str
    extracted_text: str
    ocr_confidence: float
    runtime_seconds: float
    total_blocks: int
    metrics: Optional[dict] = None

    class Config:
        json_schema_extra = {
            "example": {
                "filename": "resume.pdf",
                "cv_type": "ATS",
                "classification_confidence": 0.85,
                "ocr_engine": "EasyOCR",
                "extracted_text": "John Doe\nSoftware Engineer\n...",
                "ocr_confidence": 0.92,
                "runtime_seconds": 2.35,
                "total_blocks": 15,
                "metrics": {
                    "cer": 0.05,
                    "wer": 0.12,
                },
            }
        }


class OCRWithMetricsResponse(BaseModel):
    """Response model for CV reading with ground truth comparison"""
    filename: str
    cv_type: Literal["ATS", "Creative"]
    classification_confidence: float
    ocr_engine: str
    extracted_text: str
    ocr_confidence: float
    runtime_seconds: float
    total_blocks: int
    cer: float
    wer: float

    class Config:
        json_schema_extra = {
            "example": {
                "filename": "resume.pdf",
                "cv_type": "ATS",
                "classification_confidence": 0.85,
                "ocr_engine": "EasyOCR",
                "extracted_text": "John Doe\nSoftware Engineer\n...",
                "ocr_confidence": 0.92,
                "runtime_seconds": 2.35,
                "total_blocks": 15,
                "cer": 0.05,
                "wer": 0.12,
            }
        }


class OCRTestResponse(BaseModel):
    """Response model for testing individual OCR engines directly"""
    filename: str
    ocr_engine: str
    extracted_text: str
    ocr_confidence: float
    runtime_seconds: float
    total_blocks: int
    cer: float
    wer: float

    class Config:
        json_schema_extra = {
            "example": {
                "filename": "resume.pdf",
                "ocr_engine": "Tesseract",
                "extracted_text": "John Doe\nSoftware Engineer\n...",
                "ocr_confidence": 0.88,
                "runtime_seconds": 1.52,
                "total_blocks": 12,
                "cer": 0.07,
                "wer": 0.15,
            }
        }
