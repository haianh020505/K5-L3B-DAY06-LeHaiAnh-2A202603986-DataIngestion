"""Read and normalize source data."""
import re
import unicodedata
from pathlib import Path

from pypdf import PdfReader

from .models import DocumentPage

# OpenStax running header/footer lines, e.g. "98     4 • Decisions", "4.1 • Boolean values     87".
_BOILERPLATE = [
    re.compile(r"^Access for free at openstax\.org$"),
    re.compile(r"^\d+ +\d+ • .+$"),
    re.compile(r"^\d+\.\d+ • .+ +\d+$"),
]


def load_pdf(path: Path, document_id: str, source: str, pages: list[int]) -> tuple[list[DocumentPage], list[dict]]:
    """Extract the requested 1-based physical pages; return pages plus an error log.

    Pages with no extractable text are logged and skipped, never silently dropped.
    """
    reader = PdfReader(path)
    out, errors = [], []
    for page_no in pages:
        try:
            raw = reader.pages[page_no - 1].extract_text() or ""
        except Exception as exc:  # keep going; the log records which page failed
            errors.append({"page": page_no, "error": repr(exc)})
            continue
        text = normalize_text(raw)
        if not text:
            errors.append({"page": page_no, "error": "empty after extraction/normalization"})
            continue
        out.append(DocumentPage(document_id, source, page_no, text, {"raw_chars": len(raw)}))
    return out, errors


def normalize_text(text: str) -> str:
    """NFKC-normalize, drop running header/footer, collapse spaces; keep line breaks."""
    text = unicodedata.normalize("NFKC", text)
    lines = []
    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        if line and not any(p.match(line) for p in _BOILERPLATE):
            lines.append(line)
    return "\n".join(lines)
