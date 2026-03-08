from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.schemas.cv import ClassificationResponse, OCRResponse, OCRWithMetricsResponse
from app.services.classifier import CVClassifier
from app.services.ocr import OCREngine
from app.services.metrics import calculate_cer, calculate_wer
from app.utils.file_handler import save_upload_file, cleanup_file
from app.config import settings
from typing import Optional
import os

router = APIRouter()
classifier = CVClassifier()
ocr_engine = OCREngine()


@router.post("/classify", response_model=ClassificationResponse)
async def classify_cv(file: UploadFile = File(...)):
    """
    Classify a CV as ATS or Creative.

    - **file**: CV file (PDF, PNG, JPG, JPEG)

    Returns classification result with confidence score.
    """
    # Validate file extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Allowed: {settings.ALLOWED_EXTENSIONS}",
        )

    file_path = await save_upload_file(file)

    try:
        result = classifier.classify(file_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup_file(file_path)


@router.post("/read", response_model=OCRResponse)
async def read_cv(file: UploadFile = File(...)):
    """
    Classify and read a CV using the appropriate OCR engine.

    - **file**: CV file (PDF, PNG, JPG, JPEG)
    - ATS CVs → EasyOCR
    - Creative CVs → PaddleOCR

    Returns extracted text, OCR confidence, and runtime.
    """
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Allowed: {settings.ALLOWED_EXTENSIONS}",
        )

    file_path = await save_upload_file(file)

    try:
        # Step 1: Classify
        classification = classifier.classify(file_path)

        # Step 2: Read with appropriate OCR engine
        ocr_result = ocr_engine.read(file_path, classification.cv_type)

        return OCRResponse(
            filename=classification.filename,
            cv_type=classification.cv_type,
            classification_confidence=classification.confidence,
            ocr_engine=ocr_result["engine"],
            extracted_text=ocr_result["text"],
            ocr_confidence=ocr_result["confidence"],
            runtime_seconds=ocr_result["runtime"],
            total_blocks=ocr_result["total_blocks"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup_file(file_path)


@router.post("/read-with-metrics", response_model=OCRWithMetricsResponse)
async def read_cv_with_metrics(
    file: UploadFile = File(...),
    ground_truth: str = Form(...),
):
    """
    Classify, read a CV, and calculate CER/WER against ground truth text.

    - **file**: CV file (PDF, PNG, JPG, JPEG)
    - **ground_truth**: The expected/correct text content of the CV

    Returns extracted text, CER, WER, and runtime.
    """
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Allowed: {settings.ALLOWED_EXTENSIONS}",
        )

    file_path = await save_upload_file(file)

    try:
        # Step 1: Classify
        classification = classifier.classify(file_path)

        # Step 2: Read with appropriate OCR engine
        ocr_result = ocr_engine.read(file_path, classification.cv_type)

        # Step 3: Calculate metrics
        cer = calculate_cer(ground_truth, ocr_result["text"])
        wer = calculate_wer(ground_truth, ocr_result["text"])

        return OCRWithMetricsResponse(
            filename=classification.filename,
            cv_type=classification.cv_type,
            classification_confidence=classification.confidence,
            ocr_engine=ocr_result["engine"],
            extracted_text=ocr_result["text"],
            ocr_confidence=ocr_result["confidence"],
            runtime_seconds=ocr_result["runtime"],
            total_blocks=ocr_result["total_blocks"],
            cer=cer,
            wer=wer,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup_file(file_path)
