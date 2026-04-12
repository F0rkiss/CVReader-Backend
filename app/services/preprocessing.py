from __future__ import annotations

from typing import Final, Literal, TypedDict

import cv2
import numpy as np


ColorMode = Literal["auto", "rgb", "bgr"]


class PreprocessingMetadata(TypedDict):
    resized: bool
    grayscale: bool
    contrast_enhanced: bool


class PreprocessForOCRResult(TypedDict):
    image: np.ndarray
    metadata: PreprocessingMetadata


_TARGET_MIN_LONG_SIDE_PX: Final[int] = 1500
_TARGET_MAX_LONG_SIDE_PX: Final[int] = 2500

_ANALYSIS_MAX_LONG_SIDE_PX: Final[int] = 1200
_DESKEW_MIN_ABS_ANGLE_DEG: Final[float] = 0.5
_DESKEW_MAX_ABS_ANGLE_DEG: Final[float] = 8.0
_DESKEW_MIN_FOREGROUND_PIXELS: Final[int] = 300


def preprocess_for_ocr_image(
    image: np.ndarray,
    *,
    color_mode: ColorMode = "auto",
    min_long_side_px: int = _TARGET_MIN_LONG_SIDE_PX,
    max_long_side_px: int = _TARGET_MAX_LONG_SIDE_PX,
) -> PreprocessForOCRResult:
    if min_long_side_px < 1:
        raise ValueError("min_long_side_px must be >= 1")
    if max_long_side_px < min_long_side_px:
        raise ValueError("max_long_side_px must be >= min_long_side_px")

    image_u8 = _ensure_uint8(image)

    gray = _to_grayscale(image_u8, color_mode=color_mode)
    resized_gray, resized = _resize_to_long_side_range(
        gray,
        min_long_side_px=min_long_side_px,
        max_long_side_px=max_long_side_px,
    )

    deskewed_gray, _ = _deskew_small_angle(resized_gray)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(deskewed_gray)

    metadata: PreprocessingMetadata = {
        "resized": resized,
        "grayscale": True,
        "contrast_enhanced": True,
    }

    return {
        "image": enhanced_gray,
        "metadata": metadata,
    }


def _ensure_uint8(image: np.ndarray) -> np.ndarray:
    if image.size == 0:
        raise ValueError("Empty image")

    if image.dtype == np.uint8:
        return np.ascontiguousarray(image)

    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return np.ascontiguousarray(normalized.astype(np.uint8))


def _to_grayscale(image: np.ndarray, *, color_mode: ColorMode) -> np.ndarray:
    if image.ndim == 2:
        return image

    if image.ndim != 3:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    channels = image.shape[2]

    if channels == 3:
        if color_mode == "bgr":
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    if channels == 4:
        if color_mode == "bgr":
            bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    raise ValueError(f"Unsupported image shape: {image.shape}")


def _resize_to_long_side_range(
    image_gray: np.ndarray,
    *,
    min_long_side_px: int,
    max_long_side_px: int,
) -> tuple[np.ndarray, bool]:
    h, w = image_gray.shape[:2]
    long_side = max(h, w)

    if min_long_side_px <= long_side <= max_long_side_px:
        return image_gray, False

    if long_side < min_long_side_px:
        target_long_side = min_long_side_px
        interpolation = cv2.INTER_CUBIC
    else:
        target_long_side = max_long_side_px
        interpolation = cv2.INTER_AREA

    scale = target_long_side / float(long_side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image_gray, (new_w, new_h), interpolation=interpolation)
    return resized, True


def _downscale_for_analysis(gray: np.ndarray, max_long_side_px: int) -> np.ndarray:
    h, w = gray.shape[:2]
    long_side = max(h, w)

    if long_side <= max_long_side_px:
        return gray

    scale = max_long_side_px / float(long_side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _deskew_small_angle(image_gray: np.ndarray) -> tuple[np.ndarray, bool]:
    analysis = _downscale_for_analysis(image_gray, _ANALYSIS_MAX_LONG_SIDE_PX)

    blurred = cv2.GaussianBlur(analysis, (5, 5), 0)
    thresh = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )[1]

    points = cv2.findNonZero(thresh)
    if points is None:
        return image_gray, False

    foreground_pixels = int(points.shape[0])
    if foreground_pixels < _DESKEW_MIN_FOREGROUND_PIXELS:
        return image_gray, False

    h, w = analysis.shape[:2]
    foreground_ratio = foreground_pixels / float(max(h * w, 1))
    if foreground_ratio < 0.01 or foreground_ratio > 0.70:
        return image_gray, False

    angle = float(cv2.minAreaRect(points)[-1])
    if angle < -45.0:
        angle = -(90.0 + angle)
    else:
        angle = -angle

    abs_angle = abs(angle)
    if abs_angle < _DESKEW_MIN_ABS_ANGLE_DEG or abs_angle > _DESKEW_MAX_ABS_ANGLE_DEG:
        return image_gray, False

    src_h, src_w = image_gray.shape[:2]
    center = (src_w // 2, src_h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image_gray,
        matrix,
        (src_w, src_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    return rotated, True