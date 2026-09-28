from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from app.schemas.cv import ClassificationResponse, OCRResponse, OCRWithMetricsResponse, OCRTestResponse
from app.services.ocr import OCREngine
from app.services.classifier import CVClassifier
from app.services.metrics import calculate_cer, calculate_wer
from app.utils.file_handler import save_upload_file, cleanup_file
from app.config import settings
from typing import Callable, Optional, TypeVar
import logging
import os
import threading

logger = logging.getLogger(__name__)

router = APIRouter()
classifier = CVClassifier()
ocr_engine = OCREngine()

# OCR models are lazily initialised and not safe for concurrent inference,
# so heavy work runs one request at a time in a worker thread. This keeps the
# event loop free (e.g. /health stays responsive) while OCR is running.
_inference_lock = threading.Lock()

T = TypeVar("T")

GroundTruth = Form(..., max_length=settings.MAX_GROUND_TRUTH_CHARS)


def _validate_extension(file: UploadFile) -> None:
    file_ext = os.path.splitext(file.filename or "")[1].lower()
    if file_ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Allowed: {settings.ALLOWED_EXTENSIONS}",
        )


def _resolve_include_flag(include_preprocessed_image: bool, alias: Optional[bool]) -> bool:
    return alias if alias is not None else include_preprocessed_image


def _locked(func: Callable[[], T]) -> T:
    with _inference_lock:
        return func()


async def _run_inference(func: Callable[[], T]) -> T:
    return await run_in_threadpool(_locked, func)


def _internal_error(exc: Exception) -> HTTPException:
    logger.exception("CV processing failed")
    detail = str(exc) if settings.DEBUG else "Failed to process the CV. Check server logs for details."
    return HTTPException(status_code=500, detail=detail)


@router.post("/classify", response_model=ClassificationResponse)
async def classify_cv(file: UploadFile = File(...)):
    """
    Classify a CV as ATS or Creative.

    - **file**: CV file (PDF, PNG, JPG, JPEG, WebP, AVIF)

    Returns classification result with confidence score.
    """
    _validate_extension(file)
    file_path = await save_upload_file(file)

    try:
        classification = await _run_inference(lambda: classifier.classify(file_path))
        return classification.model_copy(update={"filename": file.filename or ""})
    except Exception as e:
        raise _internal_error(e) from e
    finally:
        cleanup_file(file_path)


@router.post("/read", response_model=OCRResponse)
async def read_cv(
    file: UploadFile = File(...),
    include_preprocessed_image: bool = Query(default=False),
    include_preprocessing_image_alias: Optional[bool] = Query(
        default=None,
        alias="include-preprocessing-image",
    ),
):
    """
    Classify and read a CV using the appropriate OCR engine.

    - **file**: CV file (PDF, PNG, JPG, JPEG, WebP, AVIF)
    - ATS CVs → EasyOCR
    - Creative CVs → PaddleOCR

    Returns extracted text, OCR confidence, and runtime.
    """
    _validate_extension(file)
    file_path = await save_upload_file(file)
    include_flag = _resolve_include_flag(include_preprocessed_image, include_preprocessing_image_alias)

    def work():
        classification = classifier.classify(file_path)
        ocr_result = ocr_engine.read(
            file_path,
            classification.cv_type,
            include_preprocessed_image=include_flag,
        )
        return classification, ocr_result

    try:
        classification, ocr_result = await _run_inference(work)

        return OCRResponse(
            filename=file.filename or "",
            cv_type=classification.cv_type,
            classification_confidence=classification.confidence,
            ocr_engine=ocr_result["engine"],
            extracted_text=ocr_result["text"],
            ocr_confidence=ocr_result["confidence"],
            runtime_seconds=ocr_result["runtime"],
            total_blocks=ocr_result["total_blocks"],
            preprocessing_metadata=ocr_result.get("preprocessing_metadata"),
        )
    except Exception as e:
        raise _internal_error(e) from e
    finally:
        cleanup_file(file_path)


@router.post("/read-with-metrics", response_model=OCRWithMetricsResponse)
async def read_cv_with_metrics(
    file: UploadFile = File(...),
    ground_truth: str = GroundTruth,
    include_preprocessed_image: bool = Query(default=False),
    include_preprocessing_image_alias: Optional[bool] = Query(
        default=None,
        alias="include-preprocessing-image",
    ),
):
    """
    Classify, read a CV, and calculate CER/WER against ground truth text.

    - **file**: CV file (PDF, PNG, JPG, JPEG, WebP, AVIF)
    - **ground_truth**: The expected/correct text content of the CV

    Returns extracted text, CER, WER, and runtime.
    """
    _validate_extension(file)
    file_path = await save_upload_file(file)
    include_flag = _resolve_include_flag(include_preprocessed_image, include_preprocessing_image_alias)

    def work():
        classification = classifier.classify(file_path)
        ocr_result = ocr_engine.read(
            file_path,
            classification.cv_type,
            include_preprocessed_image=include_flag,
        )
        cer = calculate_cer(ground_truth, ocr_result["text"])
        wer = calculate_wer(ground_truth, ocr_result["text"])
        return classification, ocr_result, cer, wer

    try:
        classification, ocr_result, cer, wer = await _run_inference(work)

        return OCRWithMetricsResponse(
            filename=file.filename or "",
            cv_type=classification.cv_type,
            classification_confidence=classification.confidence,
            ocr_engine=ocr_result["engine"],
            extracted_text=ocr_result["text"],
            ocr_confidence=ocr_result["confidence"],
            runtime_seconds=ocr_result["runtime"],
            total_blocks=ocr_result["total_blocks"],
            cer=cer,
            wer=wer,
            preprocessing_metadata=ocr_result.get("preprocessing_metadata"),
        )
    except Exception as e:
        raise _internal_error(e) from e
    finally:
        cleanup_file(file_path)


async def _run_engine_test(
    file: UploadFile,
    ground_truth: str,
    include_flag: bool,
    read_func: Callable[..., dict],
) -> OCRTestResponse:
    _validate_extension(file)
    file_path = await save_upload_file(file)

    def work():
        ocr_result = read_func(file_path, include_preprocessed_image=include_flag)
        cer = calculate_cer(ground_truth, ocr_result["text"])
        wer = calculate_wer(ground_truth, ocr_result["text"])
        return ocr_result, cer, wer

    try:
        ocr_result, cer, wer = await _run_inference(work)

        return OCRTestResponse(
            filename=file.filename or "",
            ocr_engine=ocr_result["engine"],
            extracted_text=ocr_result["text"],
            ocr_confidence=ocr_result["confidence"],
            runtime_seconds=ocr_result["runtime"],
            total_blocks=ocr_result["total_blocks"],
            cer=cer,
            wer=wer,
            preprocessing_metadata=ocr_result.get("preprocessing_metadata"),
        )
    except Exception as e:
        raise _internal_error(e) from e
    finally:
        cleanup_file(file_path)


@router.post("/test/tesseract", response_model=OCRTestResponse)
async def test_tesseract(
    file: UploadFile = File(...),
    ground_truth: str = GroundTruth,
    include_preprocessed_image: bool = Query(default=False),
    include_preprocessing_image_alias: Optional[bool] = Query(
        default=None,
        alias="include-preprocessing-image",
    ),
):
    """
    Test Tesseract OCR directly on a CV file.

    - **file**: CV file (PDF, PNG, JPG, JPEG, WebP, AVIF)
    - **ground_truth**: The expected/correct text content of the CV

    Returns extracted text, CER, WER, and runtime.
    """
    return await _run_engine_test(
        file,
        ground_truth,
        _resolve_include_flag(include_preprocessed_image, include_preprocessing_image_alias),
        ocr_engine.read_with_tesseract,
    )


@router.post("/test/easyocr", response_model=OCRTestResponse)
async def test_easyocr(
    file: UploadFile = File(...),
    ground_truth: str = GroundTruth,
    include_preprocessed_image: bool = Query(default=False),
    include_preprocessing_image_alias: Optional[bool] = Query(
        default=None,
        alias="include-preprocessing-image",
    ),
):
    """
    Test EasyOCR directly on a CV file.

    - **file**: CV file (PDF, PNG, JPG, JPEG, WebP, AVIF)
    - **ground_truth**: The expected/correct text content of the CV

    Returns extracted text, CER, WER, and runtime.
    """
    return await _run_engine_test(
        file,
        ground_truth,
        _resolve_include_flag(include_preprocessed_image, include_preprocessing_image_alias),
        ocr_engine.read_with_easyocr,
    )


@router.post("/test/paddleocr", response_model=OCRTestResponse)
async def test_paddleocr(
    file: UploadFile = File(...),
    ground_truth: str = GroundTruth,
    include_preprocessed_image: bool = Query(default=False),
    include_preprocessing_image_alias: Optional[bool] = Query(
        default=None,
        alias="include-preprocessing-image",
    ),
):
    """
    Test PaddleOCR directly on a CV file.

    - **file**: CV file (PDF, PNG, JPG, JPEG, WebP, AVIF)
    - **ground_truth**: The expected/correct text content of the CV

    Returns extracted text, CER, WER, and runtime.
    """
    return await _run_engine_test(
        file,
        ground_truth,
        _resolve_include_flag(include_preprocessed_image, include_preprocessing_image_alias),
        ocr_engine.read_with_paddleocr,
    )
