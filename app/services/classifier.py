import cv2
import numpy as np
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image
from app.schemas.cv import ClassificationResponse


class CVClassifier:
    """
    Lightweight classifier to determine if a CV is ATS or Creative.
    Uses OpenCV image analysis - fast and cheap, no heavy models.
    """

    def __init__(self):
        pass  # No heavy models to load

    def _load_image(self, file_path: str) -> np.ndarray:
        """Load image from file path (supports PDF and images)"""
        file_path = Path(file_path)

        if file_path.suffix.lower() == '.pdf':
            images = convert_from_path(str(file_path), first_page=1, last_page=1)
            if images:
                return cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2BGR)
            raise ValueError("Could not convert PDF to image")
        else:
            image = cv2.imread(str(file_path))
            if image is None:
                raise ValueError(f"Could not load image from {file_path}")
            return image

    def _analyze_colors(self, image: np.ndarray) -> dict:
        """Analyze color complexity of the CV"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 
        color_std = float(np.std(hsv[:, :, 0]))

        # Check if image is mostly grayscale vs colorful
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        color_diff = float(np.mean(np.abs(image.astype(float) - gray_3ch.astype(float))))

        return {
            "color_variance": round(color_std, 2),
            "color_diff": round(color_diff, 2),
            "is_colorful": color_diff > 10,
        }

    def _analyze_edges(self, image: np.ndarray) -> dict:
        """Analyze edge complexity and detect columns"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.sum(edges > 0) / edges.size)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
        vertical_lines = 0
        horizontal_lines = 0

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                if angle < 10 or angle > 170:
                    horizontal_lines += 1
                elif 80 < angle < 100:
                    vertical_lines += 1

        return {
            "edge_density": round(edge_density, 4),
            "vertical_lines": vertical_lines,
            "horizontal_lines": horizontal_lines,
            "has_multiple_columns": vertical_lines > 2,
        }

    def _analyze_whitespace(self, image: np.ndarray) -> dict:
        """Analyze whitespace distribution"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        whitespace_ratio = float(np.sum(binary == 255) / binary.size)

        # Divide image into grid and check whitespace variance
        h, w = binary.shape
        grid = 10
        cell_h, cell_w = h // grid, w // grid
        ws = []
        for i in range(grid):
            for j in range(grid):
                cell = binary[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w]
                ws.append(np.sum(cell == 255) / cell.size)

        layout_variance = float(np.std(ws))

        return {
            "whitespace_ratio": round(whitespace_ratio, 2),
            "layout_variance": round(layout_variance, 4),
            "is_irregular": layout_variance > 0.15,
        }

    def classify(self, file_path: str) -> ClassificationResponse:
        """
        Classify a CV as ATS or Creative using OpenCV image analysis.

        ATS: grayscale, simple edges, uniform whitespace, single column
        Creative: colorful, complex edges, irregular whitespace, multi-column
        """
        image = self._load_image(file_path)

        color_info = self._analyze_colors(image)
        edge_info = self._analyze_edges(image)
        ws_info = self._analyze_whitespace(image)

        # Calculate complexity score (max 1.0)
        score = 0.0

        if color_info["is_colorful"]:
            score += 0.15
        if color_info["color_diff"] > 20:
            score += 0.15

        if edge_info["has_multiple_columns"]:
            score += 0.2
        if edge_info["edge_density"] > 0.1:
            score += 0.2

        if ws_info["is_irregular"]:
            score += 0.2

        # ATS: 0.0 - 0.7, Creative: 0.7+
        is_creative = score >= 0.7
        cv_type = "Creative" if is_creative else "ATS"

        if is_creative:
            confidence = min(0.95, 0.5 + (score - 0.7) * 1.5)
        else:
            confidence = min(0.95, 0.5 + (0.7 - score) * 1.5)

        return ClassificationResponse(
            filename=Path(file_path).name,
            cv_type=cv_type,
            confidence=round(confidence, 2),
            details={
                "complexity_score": round(score, 2),
                "is_colorful": color_info["is_colorful"],
                "color_diff": color_info["color_diff"],
                "has_multiple_columns": edge_info["has_multiple_columns"],
                "edge_density": edge_info["edge_density"],
                "whitespace_ratio": ws_info["whitespace_ratio"],
                "layout_variance": ws_info["layout_variance"],
            },
        )
