from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


try:
    import pdfplumber  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    pdfplumber = None


@dataclass(frozen=True)
class PDFTextLayout:
    text: str
    page_width: float
    line_x_centers: list[float]

    @property
    def total_lines(self) -> int:
        return len(self.line_x_centers)


def extract_pdf_text_layout(
    file_path: str,
    *,
    max_pages: int = 1,
    line_merge_y_tolerance: float = 3.0,
) -> Optional[PDFTextLayout]:
    """Extract text + lightweight layout (line x-centers) from a PDF.

    Returns None when:
    - pdfplumber isn't installed,
    - the file isn't a PDF,
    - no extractable text is found.

    Notes:
    - This is meant to be much faster than rendering + OCR for text-based PDFs.
    - Only the first `max_pages` are scanned, matching current behavior which
      renders/OCRs the first page only.
    """

    if pdfplumber is None:
        return None

    path = Path(file_path)
    if path.suffix.lower() != ".pdf":
        return None

    if max_pages < 1:
        raise ValueError("max_pages must be >= 1")

    try:
        with pdfplumber.open(str(path)) as pdf:
            if not pdf.pages:
                return None

            all_text_lines: list[str] = []
            all_x_centers: list[float] = []
            page_width: Optional[float] = None

            for page in pdf.pages[:max_pages]:
                page_width = float(page.width)

                words = page.extract_words(
                    keep_blank_chars=False,
                    use_text_flow=True,
                )

                if not words:
                    continue

                # Sort words top-to-bottom, then left-to-right.
                words_sorted = sorted(
                    words,
                    key=lambda w: (
                        float(w.get("top", 0.0)),
                        float(w.get("x0", 0.0)),
                    ),
                )

                lines: list[list[dict]] = []
                current: list[dict] = []
                current_top: Optional[float] = None

                for word in words_sorted:
                    top = float(word.get("top", 0.0))
                    if current_top is None:
                        current = [word]
                        current_top = top
                        continue

                    if abs(top - current_top) <= line_merge_y_tolerance:
                        current.append(word)
                        # Stabilize drift by updating to the running mean.
                        current_top = (current_top * (len(current) - 1) + top) / len(
                            current
                        )
                        continue

                    lines.append(current)
                    current = [word]
                    current_top = top

                if current:
                    lines.append(current)

                for line_words in lines:
                    line_words_sorted = sorted(
                        line_words,
                        key=lambda w: float(w.get("x0", 0.0)),
                    )

                    # Split into segments when a large horizontal gap appears.
                    # This helps prevent merging two columns that share the same y.
                    gap_threshold = max(24.0, float(page_width) * 0.07)

                    segments: list[list[dict]] = []
                    current_segment: list[dict] = []
                    prev_x1: Optional[float] = None

                    for word in line_words_sorted:
                        x0_word = float(word.get("x0", 0.0))
                        x1_word = float(word.get("x1", 0.0))

                        if prev_x1 is not None and (x0_word - prev_x1) > gap_threshold:
                            if current_segment:
                                segments.append(current_segment)
                            current_segment = [word]
                        else:
                            current_segment.append(word)

                        prev_x1 = x1_word

                    if current_segment:
                        segments.append(current_segment)

                    for segment in segments:
                        texts = [str(w.get("text", "")).strip() for w in segment]
                        texts = [t for t in texts if t]
                        if not texts:
                            continue

                        x0 = min(float(w.get("x0", 0.0)) for w in segment)
                        x1 = max(float(w.get("x1", 0.0)) for w in segment)
                        all_x_centers.append((x0 + x1) / 2.0)
                        all_text_lines.append(" ".join(texts))

            if not all_text_lines or page_width is None:
                return None

            text = "\n".join(all_text_lines).strip()
            if not text:
                return None

            return PDFTextLayout(
                text=text,
                page_width=float(page_width),
                line_x_centers=all_x_centers,
            )

    except Exception:
        return None
