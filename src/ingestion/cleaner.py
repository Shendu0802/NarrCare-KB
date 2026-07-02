"""Text cleaning, denoising, and quality scoring."""
import re
from dataclasses import dataclass
from src.ingestion.parser import Page


@dataclass
class CleanedPage:
    """A cleaned page with quality metadata."""
    page_number: int
    text: str
    parse_method: str
    quality_score: float
    quality_flags: list[str]


class TextCleaner:
    """Cleans extracted text and assigns quality scores."""

    BASE_SCORE_TEXT = 1.0
    BASE_SCORE_OCR = 0.7
    QUALITY_THRESHOLD = 0.5

    PENALTIES = {
        "too_short": 0.3,
        "table_of_contents": 0.9,
        "copyright_page": 0.9,
        "isolated_characters": 0.4,
        "possible_ocr_garble": 0.3,
        "duplicated": 0.2,
        "low_information_density": 0.3,
    }

    PAGE_NUM_PATTERN = re.compile(r'^\s*\d{1,4}\s*$', re.MULTILINE)
    HEADER_FOOTER_PATTERN = re.compile(r'第\s*\d+\s*页', re.MULTILINE)

    def clean(self, page: Page) -> CleanedPage:
        """Clean a page: remove headers/footers, then score quality."""
        text = self._remove_headers_footers(page.text)
        score, flags = self.score_quality(text, page.parse_method)
        return CleanedPage(
            page_number=page.page_number, text=text,
            parse_method=page.parse_method, quality_score=score, quality_flags=flags,
        )

    def _remove_headers_footers(self, text: str) -> str:
        """Remove common header/footer patterns from text."""
        text = self.PAGE_NUM_PATTERN.sub('', text)
        text = self.HEADER_FOOTER_PATTERN.sub('', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def score_quality(self, text: str, parse_method: str) -> tuple[float, list[str]]:
        """Score text quality and return (score, flags)."""
        base = self.BASE_SCORE_TEXT if parse_method == "text_layer" else self.BASE_SCORE_OCR
        flags = []
        score = base

        if len(text) < 20:
            flags.append("too_short")
            score -= self.PENALTIES["too_short"]

        if self._is_toc(text):
            flags.append("table_of_contents")
            score -= self.PENALTIES["table_of_contents"]

        if self._is_copyright(text):
            flags.append("copyright_page")
            score -= self.PENALTIES["copyright_page"]

        if self._is_garbled(text):
            flags.append("possible_ocr_garble")
            score -= self.PENALTIES["possible_ocr_garble"]

        if self._has_isolated_chars(text):
            flags.append("isolated_characters")
            score -= self.PENALTIES["isolated_characters"]

        if self._is_low_density(text):
            flags.append("low_information_density")
            score -= self.PENALTIES["low_information_density"]

        return max(0.0, score), flags

    def _is_toc(self, text: str) -> bool:
        toc_indicators = ["目录", "Contents", "目  录"]
        dot_line_pattern = re.search(r'\.{4,}', text)
        has_toc_keyword = any(indicator in text for indicator in toc_indicators)
        return has_toc_keyword and dot_line_pattern is not None

    def _is_copyright(self, text: str) -> bool:
        indicators = ["ISBN", "CIP", "版权", "Copyright", "出版社", "印次", "版次"]
        return sum(1 for ind in indicators if ind in text) >= 2

    def _is_garbled(self, text: str) -> bool:
        if len(text) < 10:
            return False
        unusual = sum(1 for c in text
                      if ord(c) > 0x4E00
                      and ord(c) not in range(0x4E00, 0x9FFF)
                      and ord(c) not in range(0x3000, 0x303F)
                      and ord(c) not in range(0xFF00, 0xFFEF)
                      and ord(c) not in range(0x0020, 0x007F))
        return (unusual / len(text)) > 0.3

    def _has_isolated_chars(self, text: str) -> bool:
        chars = [c for c in text if '一' <= c <= '鿿']
        return len(chars) < 5 and len(text.strip()) > 0

    def _is_low_density(self, text: str) -> bool:
        if len(text) < 30:
            return False
        non_info = sum(1 for c in text if c in ' \t\n\r，。、；：""''！？…—　')
        return (non_info / len(text)) > 0.7

    def determine_status(self, quality_score: float, desired_status: str) -> str:
        if quality_score >= self.QUALITY_THRESHOLD:
            return desired_status
        return "quarantined"
