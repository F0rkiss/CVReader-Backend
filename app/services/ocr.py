import time
import numpy as np
import cv2
import easyocr
import pytesseract
from paddleocr import PaddleOCR
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image
import platform

# Set Tesseract path for Windows
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class OCREngine:
    """
    OCR Engine that uses:
    - EasyOCR for ATS CVs (simple layout)
    - PaddleOCR for Creative CVs (complex layout)
    """

    def __init__(self):
        # Lazy loading - only initialize when first needed
        self._easyocr_reader = None
        self._paddleocr_reader = None

    @property
    def easyocr_reader(self):
        """Lazy load EasyOCR"""
        if self._easyocr_reader is None:
            self._easyocr_reader = easyocr.Reader(['en'], gpu=True)
        return self._easyocr_reader

    @property
    def paddleocr_reader(self):
        """Lazy load PaddleOCR"""
        if self._paddleocr_reader is None:
            self._paddleocr_reader = PaddleOCR(
                use_angle_cls=True,
                lang='en',
                use_gpu=True,
                show_log=False,
            )
        return self._paddleocr_reader

    def _load_image(self, file_path: str) -> np.ndarray:
        """Load image from file path (supports PDF and images)"""
        file_path = Path(file_path)

        if file_path.suffix.lower() == '.pdf':
            images = convert_from_path(str(file_path), first_page=1, last_page=1)
            if images:
                return np.array(images[0])
            raise ValueError("Could not convert PDF to image")
        else:
            image = Image.open(file_path)
            return np.array(image.convert('RGB'))

    def read_with_easyocr(self, file_path: str) -> dict:
        """
        Read CV text using EasyOCR (for ATS CVs).
        Returns extracted text, confidence, and runtime.
        """
        image = self._load_image(file_path)

        start_time = time.time()
        results = self.easyocr_reader.readtext(image)
        runtime = time.time() - start_time

        # Extract text and confidence
        texts = []
        confidences = []
        for (bbox, text, conf) in results:
            texts.append(text)
            confidences.append(conf)

        extracted_text = "\n".join(texts)
        avg_confidence = float(np.mean(confidences)) if confidences else 0.0

        return {
            "text": extracted_text,
            "confidence": round(avg_confidence, 4),
            "runtime": round(runtime, 4),
            "engine": "EasyOCR",
            "total_blocks": len(results),
        }

    def read_with_paddleocr(self, file_path: str) -> dict:
        """
        Read CV text using PaddleOCR (for Creative CVs).
        Returns extracted text, confidence, and runtime.
        """
        image = self._load_image(file_path)

        start_time = time.time()
        results = self.paddleocr_reader.ocr(image, cls=True)
        runtime = time.time() - start_time

        # Extract text and confidence
        texts = []
        confidences = []

        if results and results[0]:
            for line in results[0]:
                text = line[1][0]
                conf = line[1][1]
                texts.append(text)
                confidences.append(conf)

        extracted_text = "\n".join(texts)
        avg_confidence = float(np.mean(confidences)) if confidences else 0.0

        return {
            "text": extracted_text,
            "confidence": round(avg_confidence, 4),
            "runtime": round(runtime, 4),
            "engine": "PaddleOCR",
            "total_blocks": len(texts),
        }

    def read_with_tesseract(self, file_path: str) -> dict:
        """
        Read CV text using Tesseract OCR.
        Returns extracted text, confidence, and runtime.
        """
        image = self._load_image(file_path)

        start_time = time.time()
        # Get detailed data for confidence scores
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        runtime = time.time() - start_time

        # Extract text and confidence from words with confidence > -1
        texts = []
        confidences = []
        for i, conf in enumerate(data["conf"]):
            if int(conf) > -1:
                word = data["text"][i].strip()
                if word:
                    texts.append(word)
                    confidences.append(float(conf) / 100.0)

        extracted_text = " ".join(texts)
        avg_confidence = float(np.mean(confidences)) if confidences else 0.0

        return {
            "text": extracted_text,
            "confidence": round(avg_confidence, 4),
            "runtime": round(runtime, 4),
            "engine": "Tesseract",
            "total_blocks": len(texts),
        }

    def read(self, file_path: str, cv_type: str) -> dict:
        """
        Read CV using the appropriate OCR engine based on CV type.

        Args:
            file_path: Path to the CV file
            cv_type: "ATS" or "Creative"

        Returns:
            dict with text, confidence, runtime, engine, total_blocks
        """
        if cv_type == "Creative":
            return self.read_with_paddleocr(file_path)
        else:
            return self.read_with_easyocr(file_path)
