"""Semantic chunking — splits cleaned text by semantic boundaries, not fixed char counts."""
import uuid
from src.ingestion.cleaner import CleanedPage


class SemanticChunker:
    """Splits documents into semantically coherent chunks with overlap."""

    def __init__(self, min_chars: int = 300, max_chars: int = 800, overlap_chars: int = 100):
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    def _make_id(self, source_type_short: str = "doc") -> str:
        short_uuid = uuid.uuid4().hex[:8]
        return f"ku_{source_type_short}_{short_uuid}"

    def chunk(
        self, cleaned_pages: list[CleanedPage],
        source_title: str = "", source_uri: str = "", source_type: str = "pdf_book",
    ) -> list[dict]:
        """Split cleaned pages into semantic chunks. Returns list of chunk dicts."""
        segments = self._split_by_boundaries(cleaned_pages)
        merged = self._merge_short_segments(segments)

        final_segments = []
        for seg in merged:
            if len(seg["text"]) > self.max_chars:
                final_segments.extend(self._split_long_segment(seg))
            else:
                final_segments.append(seg)

        chunks = []
        source_type_short = source_type.replace("_", "")[:12]
        prev_text = ""
        for seg in final_segments:
            text = seg["text"]
            if prev_text and self.overlap_chars > 0:
                overlap = prev_text[-self.overlap_chars:]
                text = overlap + "\n" + text

            chunks.append({
                "id": self._make_id(source_type_short),
                "unit_type": "semantic_chunk",
                "source_type": source_type,
                "text": text,
                "title": self._extract_title(text),
                "source_title": source_title,
                "source_uri": source_uri,
                "page_start": seg.get("page_start"),
                "page_end": seg.get("page_end"),
                "parse_method": seg.get("parse_method", "text_layer"),
            })
            prev_text = seg["text"]

        return chunks

    def _split_by_boundaries(self, cleaned_pages: list[CleanedPage]) -> list[dict]:
        segments = []
        for page in cleaned_pages:
            paragraphs = page.text.split("\n\n")
            for para in paragraphs:
                para = para.strip()
                if para:
                    segments.append({
                        "text": para, "page_start": page.page_number,
                        "page_end": page.page_number, "parse_method": page.parse_method,
                    })
        return segments

    def _merge_short_segments(self, segments: list[dict]) -> list[dict]:
        merged = []
        i = 0
        while i < len(segments):
            current = segments[i]
            while len(current["text"]) < self.min_chars and i + 1 < len(segments):
                i += 1
                current["text"] += "\n" + segments[i]["text"]
                current["page_end"] = segments[i]["page_end"]
            merged.append(current)
            i += 1
        return merged

    def _split_long_segment(self, seg: dict) -> list[dict]:
        text = seg["text"]
        if len(text) <= self.max_chars:
            return [seg]

        parts = []
        current = ""
        for char in text:
            current += char
            if char in "。！？\n" and len(current) >= self.min_chars:
                parts.append(current.strip())
                current = ""
        if current.strip():
            if parts and len(current) < self.min_chars:
                parts[-1] += current
            else:
                parts.append(current.strip())

        return [{**seg, "text": p} for p in parts if p]

    @staticmethod
    def _extract_title(text: str) -> str:
        first_line = text.split("\n")[0].strip()
        if len(first_line) <= 50:
            return first_line
        return first_line[:47] + "..."
