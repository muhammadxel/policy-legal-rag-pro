import io
import re
from pypdf import PdfReader
from docx import Document


def clean_text(text: str) -> str:
    """Normalize whitespace and strip broken line wraps."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_with_overlap(text: str, chunk_size: int = 350, overlap: int = 50) -> list[str]:
    """Split text into overlapping word chunks to preserve boundary context."""
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def extract_pdf_chunks(file_bytes: bytes, filename: str) -> tuple[list[dict], dict]:
    """Extract text page-by-page from a PDF and attach page-level metadata."""
    reader = PdfReader(io.BytesIO(file_bytes))
    records = []
    total_words = 0

    for page_idx, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        cleaned = clean_text(raw_text)
        if not cleaned:
            continue

        total_words += len(cleaned.split())
        page_chunks = chunk_with_overlap(cleaned)

        for chunk_idx, chunk_text in enumerate(page_chunks):
            records.append({
                "id": f"{filename}_p{page_idx}_c{chunk_idx}",
                "text": chunk_text,
                "metadata": {
                    "source": filename,
                    "page": page_idx,
                    "chunk_index": chunk_idx,
                    "file_type": "PDF",
                },
            })

    stats = {
        "filename": filename,
        "pages_or_sections": len(reader.pages),
        "words": total_words,
        "chunks": len(records),
    }
    return records, stats


def extract_docx_chunks(file_bytes: bytes, filename: str) -> tuple[list[dict], dict]:
    """Extract paragraphs from a DOCX file and group into logical sections."""
    doc = Document(io.BytesIO(file_bytes))
    paragraphs = [clean_text(p.text) for p in doc.paragraphs if clean_text(p.text)]
    full_text = "\n".join(paragraphs)
    total_words = len(full_text.split())

    chunks = chunk_with_overlap(full_text)
    records = []
    for chunk_idx, chunk_text in enumerate(chunks):
        # Estimate logical section number (roughly every 2 chunks ~ 1 page equivalent)
        logical_section = (chunk_idx // 2) + 1
        records.append({
            "id": f"{filename}_s{logical_section}_c{chunk_idx}",
            "text": chunk_text,
            "metadata": {
                "source": filename,
                "page": logical_section,
                "chunk_index": chunk_idx,
                "file_type": "DOCX",
            },
        })

    stats = {
        "filename": filename,
        "pages_or_sections": max(1, (len(chunks) + 1) // 2),
        "words": total_words,
        "chunks": len(records),
    }
    return records, stats


def process_uploaded_files(uploaded_files) -> tuple[list[dict], list[dict]]:
    """Process a batch of PDF and DOCX files into metadata-tagged records."""
    all_records = []
    all_stats = []

    for file in uploaded_files:
        file_bytes = file.getvalue()
        filename = file.name

        if filename.lower().endswith(".pdf"):
            records, stats = extract_pdf_chunks(file_bytes, filename)
        elif filename.lower().endswith(".docx"):
            records, stats = extract_docx_chunks(file_bytes, filename)
        else:
            continue

        all_records.extend(records)
        all_stats.append(stats)

    return all_records, all_stats