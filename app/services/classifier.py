import importlib
from pathlib import Path

import numpy as np
from paddleocr import PaddleOCR
from pdf2image import convert_from_path
from PIL import Image

from app.schemas.cv import ClassificationResponse

try:
    importlib.import_module("pillow_avif")
except ModuleNotFoundError:
    pass


class CVClassifier:
    """Classify CVs as ATS or Creative using PaddleOCR layout signals."""

    def __init__(self):
        self._ocr = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            use_gpu=True,
            show_log=False,
        )

    def _load_image(self, file_path: str) -> np.ndarray:
        """Load image from path (supports PDF and image formats)."""
        file = Path(file_path)

        if file.suffix.lower() == ".pdf":
            pages = convert_from_path(str(file), first_page=1, last_page=1)
            if not pages:
                raise ValueError("Could not convert PDF to image")
            return np.array(pages[0].convert("RGB"))

        return np.array(Image.open(file).convert("RGB"))

    def classify(self, file_path: str) -> ClassificationResponse:
        """Classify a CV using text block distribution from PaddleOCR output."""
        image = self._load_image(file_path)
        _, width = image.shape[:2]

        results = self._ocr.ocr(image, cls=True)
        lines = results[0] if results and results[0] else []

        if not lines:
            return ClassificationResponse(
                filename=Path(file_path).name,
                cv_type="Creative",
                confidence=0.55,
                details={
                    "reason": "no_text_detected",
                    "total_blocks": 0,
                },
            )

        confidences = []
        x_centers = []

        for line in lines:
            box = line[0]
            confidence = float(line[1][1])
            confidences.append(confidence)

            x_values = [point[0] for point in box]
            x_center = float(sum(x_values) / len(x_values))
            x_centers.append(x_center)

        total_blocks = len(lines)
        avg_confidence = float(np.mean(confidences)) if confidences else 0.0

        left_blocks = sum(1 for x in x_centers if x < 0.45 * width)
        right_blocks = sum(1 for x in x_centers if x > 0.55 * width)
        has_multiple_columns = (
            left_blocks > 0.2 * total_blocks and right_blocks > 0.2 * total_blocks
        )

        creative_score = 0.0
        if has_multiple_columns:
            creative_score += 0.6
        if avg_confidence < 0.80:
            creative_score += 0.2
        if total_blocks > 35:
            creative_score += 0.2

        cv_type = "Creative" if creative_score >= 0.6 else "ATS"
        confidence = min(0.95, 0.55 + 0.4 * abs(creative_score - 0.6))

        return ClassificationResponse(
            filename=Path(file_path).name,
            cv_type=cv_type,
            confidence=round(confidence, 2),
            details={
                "total_blocks": total_blocks,
                "avg_ocr_confidence": round(avg_confidence, 4),
                "has_multiple_columns": has_multiple_columns,
                "left_blocks": left_blocks,
                "right_blocks": right_blocks,
                "creative_score": round(creative_score, 2),
            },
        )
