"""Fixed-size and structure-aware chunking; every chunk keeps page + chapter/section metadata."""
import re

from .models import Chunk, DocumentPage

# PDF lines at least this long are hard-wrapped prose that continues on the next line.
WRAP_WIDTH = 90


def _make(page: DocumentPage, index: int, start: int, text: str) -> Chunk:
    # section active at the chunk's first character (markers come from the PDF outline)
    section = page.metadata["section_at_start"]
    for offset, name in page.metadata["section_markers"]:
        if offset <= start:
            section = name
    return Chunk(
        chunk_id=f"{page.document_id}-p{page.page:03d}-c{index:02d}",
        document_id=page.document_id, source=page.source, page=page.page, text=text,
        metadata={"chapter": page.metadata["chapter"], "section": section, "char_start": start},
    )


def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    """Overlapping character windows inside one page."""
    assert 0 <= overlap < size
    chunks, start = [], 0
    while start < len(page.text):
        piece = page.text[start:start + size].strip()
        if piece:
            chunks.append(_make(page, len(chunks), start, piece))
        if start + size >= len(page.text):
            break
        start += size - overlap
    return chunks


def _blocks(text: str) -> list[tuple[int, str]]:
    """Re-join hard-wrapped lines into paragraphs; return (offset, paragraph)."""
    blocks, offset, cur, cur_start = [], 0, [], 0
    for line in text.split("\n"):
        if not cur:
            cur_start = offset
        cur.append(line)
        if len(line) < WRAP_WIDTH:  # short line ends a paragraph (or is a heading/code/option line)
            blocks.append((cur_start, " ".join(cur)))
            cur = []
        offset += len(line) + 1
    if cur:
        blocks.append((cur_start, " ".join(cur)))
    return blocks


def _split_long(text: str, max_size: int) -> list[str]:
    """Pack sentences up to max_size; hard-cut a single sentence that is still too long."""
    parts, cur = [], ""
    for sent in re.split(r"(?<=[.?!])\s+", text):
        while len(sent) > max_size:
            parts.append(sent[:max_size])
            sent = sent[max_size:]
        if cur and len(cur) + 1 + len(sent) > max_size:
            parts.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        parts.append(cur)
    return parts


def by_paragraph(page: DocumentPage, max_size: int) -> list[Chunk]:
    """Group paragraphs up to max_size; start a new chunk at each section heading."""
    headings = {offset for offset, _ in page.metadata["section_markers"]}
    chunks, cur, cur_start = [], [], 0

    def flush():
        if cur:
            chunks.append(_make(page, len(chunks), cur_start, "\n".join(cur)))
            cur.clear()

    for offset, para in _blocks(page.text):
        if offset in headings or (cur and len("\n".join(cur)) + 1 + len(para) > max_size):
            flush()
        if len(para) > max_size:
            for piece in _split_long(para, max_size):
                cur_start = offset
                cur.append(piece)
                flush()
            continue
        if not cur:
            cur_start = offset
        cur.append(para)
    flush()
    return chunks
