import time
import base64
import re
import shutil
import numpy as np
import cv2
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image
import platform
import importlib

try:
    import pytesseract
except ModuleNotFoundError:
    pytesseract = None

from app.services.preprocessing import preprocess_for_ocr_image
from app.utils.poppler import resolve_poppler_path

try:
    importlib.import_module("pillow_avif")
except ModuleNotFoundError:
    pass

if platform.system() == "Windows" and pytesseract is not None:
    default_tesseract_path = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if default_tesseract_path.exists():
        pytesseract.pytesseract.tesseract_cmd = str(default_tesseract_path)


class OCREngine:
    """
    Website logic:
    - ATS -> EasyOCR
    - Creative -> PaddleOCR
    - fallback to an alternative only when needed (Creative path)
    """

    def __init__(self):
        self._easyocr_reader = None
        self._paddleocr_reader = None
        self._easyocr_import_error: Exception | None = None
        self._paddleocr_import_error: Exception | None = None
        self._easyocr_runtime_mode = "unknown"
        self._poppler_path = resolve_poppler_path()

    @property
    def easyocr_reader(self):
        if self._easyocr_reader is None:
            try:
                easyocr_module = importlib.import_module("easyocr")
            except ModuleNotFoundError as exc:
                self._easyocr_import_error = exc
                raise RuntimeError(
                    "EasyOCR is not installed. Install dependencies from requirements.txt "
                    "or run: pip install easyocr"
                ) from exc

            use_gpu = False
            try:
                torch_module = importlib.import_module("torch")
                cuda = getattr(torch_module, "cuda", None)
                use_gpu = bool(cuda and cuda.is_available())
            except Exception as exc:
                self._easyocr_import_error = exc
                use_gpu = False

            try:
                self._easyocr_reader = easyocr_module.Reader(["en"], gpu=use_gpu)
                self._easyocr_runtime_mode = "gpu" if use_gpu else "cpu"
            except Exception as exc:
                if use_gpu:
                    # Retry on CPU when CUDA initialization fails.
                    try:
                        self._easyocr_reader = easyocr_module.Reader(["en"], gpu=False)
                        self._easyocr_runtime_mode = "cpu"
                    except Exception as cpu_exc:
                        self._easyocr_import_error = cpu_exc
                        raise RuntimeError(
                            "EasyOCR failed to initialize on both GPU and CPU. "
                            "Reinstall easyocr/torch and retry."
                        ) from cpu_exc
                else:
                    self._easyocr_import_error = exc
                    raise RuntimeError(
                        "EasyOCR failed to initialize. "
                        "Ensure easyocr and torch are installed correctly."
                    ) from exc
        return self._easyocr_reader

    @property
    def paddleocr_reader(self):
        if self._paddleocr_reader is None:
            try:
                paddleocr_module = importlib.import_module("paddleocr")
                paddleocr_cls = paddleocr_module.PaddleOCR
                self._paddleocr_reader = paddleocr_cls(
                    use_angle_cls=True,
                    lang="en",
                    use_gpu=True,
                    show_log=False,
                )
            except Exception as exc:
                self._paddleocr_import_error = exc
                raise RuntimeError(
                    "PaddleOCR is unavailable in this environment."
                ) from exc
        return self._paddleocr_reader

    def _load_image(self, file_path: str) -> np.ndarray:
        file_path = Path(file_path)

        if file_path.suffix.lower() == ".pdf":
            images = convert_from_path(
                str(file_path),
                first_page=1,
                last_page=1,
                dpi=200,
                poppler_path=self._poppler_path,
            )
            if images:
                return np.array(images[0].convert("RGB"))
            raise ValueError("Could not convert PDF to image")

        image = Image.open(file_path)
        return np.array(image.convert("RGB"))

    def _filter_metadata(self, raw_metadata: dict) -> dict:
        return {
            "resized": bool(raw_metadata.get("resized", False)),
            "grayscale": bool(raw_metadata.get("grayscale", False)),
            "contrast_enhanced": bool(raw_metadata.get("contrast_enhanced", False)),
        }

    def _preprocess_image_for_ocr(self, image: np.ndarray) -> dict:
        try:
            result = preprocess_for_ocr_image(
                image,
                color_mode="rgb",
            )

            processed = result["image"]
            preprocessing_metadata = self._filter_metadata(result["metadata"])

            ocr_ready = processed
            if ocr_ready.ndim == 2:
                ocr_ready = cv2.cvtColor(ocr_ready, cv2.COLOR_GRAY2RGB)

            return {
                "ocr_image": np.ascontiguousarray(ocr_ready),
                "preprocessed_image": np.ascontiguousarray(processed),
                "preprocessing_metadata": preprocessing_metadata,
            }

        except Exception:
            return {
                "ocr_image": image,
                "preprocessed_image": image,
                "preprocessing_metadata": {
                    "resized": False,
                    "grayscale": False,
                    "contrast_enhanced": False,
                },
            }

    def _encode_png_base64(self, image: np.ndarray) -> str:
        if image.ndim == 3 and image.shape[2] == 3:
            to_encode = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif image.ndim == 3 and image.shape[2] == 4:
            to_encode = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
        else:
            to_encode = image

        ok, buffer = cv2.imencode(".png", to_encode)
        if not ok:
            raise ValueError("Failed to encode preprocessed image as PNG")
        return base64.b64encode(buffer.tobytes()).decode("ascii")

    def _attach_preprocessing_payload(
        self,
        result: dict,
        preprocess_data: dict,
        include_preprocessed_image: bool,
    ) -> dict:
        result["preprocessing_metadata"] = preprocess_data["preprocessing_metadata"]

        if include_preprocessed_image:
            result["preprocessed_image_png_base64"] = self._encode_png_base64(
                preprocess_data["preprocessed_image"]
            )

        return result

    def _clean_text_for_quality(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def _gibberish_ratio(self, text: str) -> float:
        clean = self._clean_text_for_quality(text)
        if not clean:
            return 1.0

        allowed_chars = sum(
            1 for ch in clean
            if ch.isalnum() or ch in " .,;:!?@#%&()[]{}+-_/\\'\""
        )
        return 1.0 - (allowed_chars / max(len(clean), 1))

    def _looks_like_bad_ocr(self, result: dict) -> bool:
        text = str(result.get("text", "")).strip()
        confidence = float(result.get("confidence", 0.0))
        total_blocks = int(result.get("total_blocks", 0))

        if confidence < 0.50:
            return True

        if len(text) < 80:
            return True

        if total_blocks <= 2:
            return True

        gibberish_ratio = self._gibberish_ratio(text)
        if gibberish_ratio > 0.22:
            return True

        return False

    def _score_result(self, result: dict) -> float:
        text = str(result.get("text", "")).strip()
        confidence = float(result.get("confidence", 0.0))
        total_blocks = int(result.get("total_blocks", 0))
        gibberish_penalty = self._gibberish_ratio(text)

        text_len_score = min(len(text) / 1500.0, 1.0)
        block_score = min(total_blocks / 40.0, 1.0)

        score = (
            confidence * 0.55
            + text_len_score * 0.20
            + block_score * 0.10
            + (1.0 - gibberish_penalty) * 0.15
        )
        return score

    def _read_easyocr_from_image(self, ocr_image: np.ndarray) -> dict:
        ocr_ready = np.ascontiguousarray(ocr_image)
        if ocr_ready.dtype != np.uint8:
            normalized = cv2.normalize(ocr_ready, None, 0, 255, cv2.NORM_MINMAX)
            ocr_ready = normalized.astype(np.uint8)

        start_time = time.time()
        results = self.easyocr_reader.readtext(ocr_ready, detail=1, paragraph=False)
        runtime = time.time() - start_time

        texts = []
        confidences = []

        for item in results:
            if len(item) < 3:
                continue
            text = str(item[1]).strip()
            conf = float(item[2])
            if text:
                texts.append(text)
                confidences.append(conf)

        extracted_text = "\n".join(texts)
        avg_confidence = float(np.mean(confidences)) if confidences else 0.0

        return {
            "text": extracted_text,
            "confidence": round(avg_confidence, 4),
            "runtime": round(runtime, 4),
            "engine": "EasyOCR",
            "total_blocks": len(texts),
        }

    def _read_paddleocr_from_image(self, ocr_image: np.ndarray) -> dict:
        start_time = time.time()
        results = self.paddleocr_reader.ocr(ocr_image, cls=True)
        runtime = time.time() - start_time

        texts = []
        confidences = []

        if results and results[0]:
            for line in results[0]:
                text = str(line[1][0]).strip()
                conf = float(line[1][1])
                if text:
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

    def _read_tesseract_from_image(self, ocr_image: np.ndarray) -> dict:
        if pytesseract is None:
            raise RuntimeError("pytesseract is not installed. Run: pip install pytesseract")

        configured_cmd = str(getattr(pytesseract.pytesseract, "tesseract_cmd", "")).strip()
        has_configured_cmd = bool(configured_cmd and Path(configured_cmd).exists())
        has_path_binary = shutil.which("tesseract") is not None

        if not (has_configured_cmd or has_path_binary):
            raise RuntimeError("Tesseract is not installed or not available in PATH.")

        start_time = time.time()
        try:
            data = pytesseract.image_to_data(
                ocr_image,
                output_type=pytesseract.Output.DICT,
            )
        except Exception as exc:
            raise RuntimeError("Tesseract is not installed or not available in PATH.") from exc
        runtime = time.time() - start_time

        texts = []
        confidences = []

        for i, conf in enumerate(data["conf"]):
            try:
                conf_value = float(conf)
            except (TypeError, ValueError):
                continue

            if conf_value > -1:
                word = str(data["text"][i]).strip()
                if word:
                    texts.append(word)
                    confidences.append(conf_value / 100.0)

        extracted_text = " ".join(texts)
        avg_confidence = float(np.mean(confidences)) if confidences else 0.0

        return {
            "text": extracted_text,
            "confidence": round(avg_confidence, 4),
            "runtime": round(runtime, 4),
            "engine": "Tesseract",
            "total_blocks": len(texts),
        }

    def read_with_easyocr(self, file_path: str, include_preprocessed_image: bool = False) -> dict:
        image = self._load_image(file_path)
        preprocess_data = self._preprocess_image_for_ocr(image)

        result = self._read_easyocr_from_image(preprocess_data["ocr_image"])

        return self._attach_preprocessing_payload(
            result,
            preprocess_data,
            include_preprocessed_image,
        )

    def read_with_paddleocr(self, file_path: str, include_preprocessed_image: bool = False) -> dict:
        image = self._load_image(file_path)
        preprocess_data = self._preprocess_image_for_ocr(image)

        result = self._read_paddleocr_from_image(preprocess_data["ocr_image"])

        return self._attach_preprocessing_payload(
            result,
            preprocess_data,
            include_preprocessed_image,
        )

    def read_with_tesseract(self, file_path: str, include_preprocessed_image: bool = False) -> dict:
        image = self._load_image(file_path)
        preprocess_data = self._preprocess_image_for_ocr(image)

        result = self._read_tesseract_from_image(preprocess_data["ocr_image"])

        return self._attach_preprocessing_payload(
            result,
            preprocess_data,
            include_preprocessed_image,
        )

    def read(
        self,
        file_path: str,
        cv_type: str,
        include_preprocessed_image: bool = False,
    ) -> dict:
        """
        Main website route:
        - ATS => EasyOCR only
        - Creative => PaddleOCR first, fallback to EasyOCR if quality is bad
        """
        # Load + preprocess once; reuse for primary/fallback to avoid re-rendering PDFs.
        image = self._load_image(file_path)
        preprocess_data = self._preprocess_image_for_ocr(image)
        ocr_image = preprocess_data["ocr_image"]

        if cv_type == "Creative":
            try:
                primary = self._read_paddleocr_from_image(ocr_image)
            except Exception:
                primary = self._read_easyocr_from_image(ocr_image)

            primary = self._attach_preprocessing_payload(
                primary,
                preprocess_data,
                include_preprocessed_image,
            )

            if self._looks_like_bad_ocr(primary):
                try:
                    fallback = self._read_easyocr_from_image(ocr_image)
                    fallback = self._attach_preprocessing_payload(
                        fallback,
                        preprocess_data,
                        include_preprocessed_image,
                    )
                    if self._score_result(fallback) > self._score_result(primary):
                        fallback["fallback_used"] = True
                        fallback["primary_engine"] = "PaddleOCR"
                        return fallback
                except Exception:
                    pass

            primary["fallback_used"] = False
            primary["primary_engine"] = "PaddleOCR"
            return primary

        primary_engine = "EasyOCR"
        ats_errors: list[str] = []
        used_original_image_for_easyocr = False

        try:
            primary = self._read_easyocr_from_image(ocr_image)
        except Exception as exc:
            ats_errors.append(f"preprocessed_image_error={exc}")
            try:
                primary = self._read_easyocr_from_image(image)
                used_original_image_for_easyocr = True
            except Exception as raw_exc:
                ats_errors.append(f"original_image_error={raw_exc}")
                init_error = str(self._easyocr_import_error) if self._easyocr_import_error else None

                detail = "; ".join(ats_errors)
                if init_error:
                    detail = f"{detail}; init_error={init_error}"
                detail = f"{detail}; runtime_mode={self._easyocr_runtime_mode}"

                raise RuntimeError(f"EasyOCR failed for ATS CV: {detail}") from raw_exc

        primary = self._attach_preprocessing_payload(
            primary,
            preprocess_data,
            include_preprocessed_image,
        )

        primary["fallback_used"] = False
        primary["primary_engine"] = primary_engine
        primary["easyocr_runtime_mode"] = self._easyocr_runtime_mode
        primary["easyocr_input_variant"] = (
            "original_image" if used_original_image_for_easyocr else "preprocessed_image"
        )
        return primary