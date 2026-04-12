import os
import time
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query

from app.config import settings
from app.schemas.cv import (
    ClassificationResponse,
    OCRResponse,
    OCRTestResponse,
    OCRWithMetricsResponse,
)
from app.services.classifier import CVClassifier
from app.services.metrics import calculate_cer, calculate_wer
from app.services.ocr import OCREngine
from app.services.pdf_extraction import PDFTextLayout, extract_pdf_text_layout
from app.utils.file_handler import cleanup_file, save_upload_file

router = APIRouter()
classifier = CVClassifier()
ocr_engine = OCREngine()


def _validate_file_extension(filename: str) -> str:
    file_ext = os.path.splitext(filename)[1].lower()
    if file_ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Allowed: {settings.ALLOWED_EXTENSIONS}",
        )
    return file_ext


def _resolve_include_preprocessed_image_flag(
    include_preprocessed_image: bool,
    include_preprocessing_image: Optional[bool],
    include_preprocessing_image_alias: Optional[bool],
) -> bool:
    if include_preprocessing_image_alias is not None:
        return include_preprocessing_image_alias
    if include_preprocessing_image is not None:
        return include_preprocessing_image
    return include_preprocessed_image


def _extract_pdf_layout_if_needed(
    file_path: str,
    file_ext: str,
) -> tuple[PDFTextLayout | None, float | None]:
    if file_ext != ".pdf":
        return None, None

    started_at = time.time()
    pdf_layout = extract_pdf_text_layout(file_path, max_pages=1)
    if pdf_layout is None:
        return None, None

    return pdf_layout, time.time() - started_at


def _classify_and_read_cv(
    file_path: str,
    file_ext: str,
    include_preprocessed_image: bool,
) -> tuple[ClassificationResponse, dict]:
    pdf_layout, pdf_layout_runtime = _extract_pdf_layout_if_needed(file_path, file_ext)

    classification = classifier.classify(
        file_path,
        pdf_layout=pdf_layout,
    )
    ocr_result = ocr_engine.read(
        file_path,
        classification.cv_type,
        include_preprocessed_image=include_preprocessed_image,
        pdf_layout=pdf_layout,
        pdf_layout_runtime=pdf_layout_runtime,
    )
    return classification, ocr_result


def _optional_preprocessed_image(
    ocr_result: dict,
    include_preprocessed_image: bool,
) -> Optional[str]:
    if not include_preprocessed_image:
        return None
    return ocr_result.get("preprocessed_image_png_base64")


def _build_ocr_test_response(
    filename: str,
    ground_truth: str,
    ocr_result: dict,
    include_preprocessed_image: bool,
) -> OCRTestResponse:
    return OCRTestResponse(
        filename=filename,
        ocr_engine=ocr_result["engine"],
        extracted_text=ocr_result["text"],
        ocr_confidence=ocr_result["confidence"],
        runtime_seconds=ocr_result["runtime"],
        total_blocks=ocr_result["total_blocks"],
        cer=calculate_cer(ground_truth, ocr_result["text"]),
        wer=calculate_wer(ground_truth, ocr_result["text"]),
        preprocessing_metadata=ocr_result["preprocessing_metadata"],
        preprocessed_image_png_base64=_optional_preprocessed_image(
            ocr_result,
            include_preprocessed_image,
        ),
    )


@router.post("/classify", response_model=ClassificationResponse)
async def classify_cv(file: UploadFile = File(...)):
    """
    Classify a CV as ATS or Creative.

    - **file**: CV file (PDF, PNG, JPG, JPEG, WebP, AVIF)

    Returns classification result with confidence score.
    """
    _validate_file_extension(file.filename)

    file_path = await save_upload_file(file)

    try:
        result = classifier.classify(file_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup_file(file_path)


@router.post("/read", response_model=OCRResponse)
async def read_cv(
    file: UploadFile = File(...),
    include_preprocessed_image: bool = Query(default=False),
    include_preprocessing_image: Optional[bool] = Query(default=None),
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
    file_ext = _validate_file_extension(file.filename)

    file_path = await save_upload_file(file)

    try:
        include_preprocessed_image_flag = _resolve_include_preprocessed_image_flag(
            include_preprocessed_image,
            include_preprocessing_image,
            include_preprocessing_image_alias,
        )

        classification, ocr_result = _classify_and_read_cv(
            file_path,
            file_ext,
            include_preprocessed_image_flag,
        )

        return OCRResponse(
            filename=classification.filename,
            cv_type=classification.cv_type,
            classification_confidence=classification.confidence,
            ocr_engine=ocr_result["engine"],
            extracted_text=ocr_result["text"],
            ocr_confidence=ocr_result["confidence"],
            runtime_seconds=ocr_result["runtime"],
            total_blocks=ocr_result["total_blocks"],
            preprocessing_metadata=ocr_result["preprocessing_metadata"],
            preprocessed_image_png_base64=_optional_preprocessed_image(
                ocr_result,
                include_preprocessed_image_flag,
            ),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup_file(file_path)


@router.post("/read-with-metrics", response_model=OCRWithMetricsResponse)
async def read_cv_with_metrics(
    file: UploadFile = File(...),
    ground_truth: str = Form(...),
    include_preprocessed_image: bool = Query(default=False),
    include_preprocessing_image: Optional[bool] = Query(default=None),
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
    file_ext = _validate_file_extension(file.filename)

    file_path = await save_upload_file(file)

    try:
        include_preprocessed_image_flag = _resolve_include_preprocessed_image_flag(
            include_preprocessed_image,
            include_preprocessing_image,
            include_preprocessing_image_alias,
        )

        classification, ocr_result = _classify_and_read_cv(
            file_path,
            file_ext,
            include_preprocessed_image_flag,
        )

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
            preprocessing_metadata=ocr_result["preprocessing_metadata"],
            preprocessed_image_png_base64=_optional_preprocessed_image(
                ocr_result,
                include_preprocessed_image_flag,
            ),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup_file(file_path)


@router.post("/test/tesseract", response_model=OCRTestResponse)
async def test_tesseract(
    file: UploadFile = File(...),
    ground_truth: str = Form(...),
    include_preprocessed_image: bool = Query(default=False),
    include_preprocessing_image: Optional[bool] = Query(default=None),
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
    _validate_file_extension(file.filename)

    file_path = await save_upload_file(file)

    try:
        include_preprocessed_image_flag = _resolve_include_preprocessed_image_flag(
            include_preprocessed_image,
            include_preprocessing_image,
            include_preprocessing_image_alias,
        )

        ocr_result = ocr_engine.read_with_tesseract(
            file_path,
            include_preprocessed_image=include_preprocessed_image_flag,
        )
        return _build_ocr_test_response(
            filename=file.filename,
            ground_truth=ground_truth,
            ocr_result=ocr_result,
            include_preprocessed_image=include_preprocessed_image_flag,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup_file(file_path)


@router.post("/test/easyocr", response_model=OCRTestResponse)
async def test_easyocr(
    file: UploadFile = File(...),
    ground_truth: str = Form(...),
    include_preprocessed_image: bool = Query(default=False),
    include_preprocessing_image: Optional[bool] = Query(default=None),
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
    _validate_file_extension(file.filename)

    file_path = await save_upload_file(file)

    try:
        include_preprocessed_image_flag = _resolve_include_preprocessed_image_flag(
            include_preprocessed_image,
            include_preprocessing_image,
            include_preprocessing_image_alias,
        )

        ocr_result = ocr_engine.read_with_easyocr(
            file_path,
            include_preprocessed_image=include_preprocessed_image_flag,
        )
        return _build_ocr_test_response(
            filename=file.filename,
            ground_truth=ground_truth,
            ocr_result=ocr_result,
            include_preprocessed_image=include_preprocessed_image_flag,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup_file(file_path)


@router.post("/test/paddleocr", response_model=OCRTestResponse)
async def test_paddleocr(
    file: UploadFile = File(...),
    ground_truth: str = Form(...),
    include_preprocessed_image: bool = Query(default=False),
    include_preprocessing_image: Optional[bool] = Query(default=None),
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
    _validate_file_extension(file.filename)

    file_path = await save_upload_file(file)

    try:
        include_preprocessed_image_flag = _resolve_include_preprocessed_image_flag(
            include_preprocessed_image,
            include_preprocessing_image,
            include_preprocessing_image_alias,
        )

        ocr_result = ocr_engine.read_with_paddleocr(
            file_path,
            include_preprocessed_image=include_preprocessed_image_flag,
        )
        return _build_ocr_test_response(
            filename=file.filename,
            ground_truth=ground_truth,
            ocr_result=ocr_result,
            include_preprocessed_image=include_preprocessed_image_flag,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup_file(file_path)
