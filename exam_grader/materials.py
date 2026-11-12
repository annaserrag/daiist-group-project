"""Page-preserving course PDF ingestion for a single Gradio session."""
import os
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PyPdfError

from .build import MODEL, digest, vector
from .retrieval import Index

BUILTIN = "Built-in ML references"
MAX_PASSAGES = 240


def extract(path, raw, source):
    try:
        pdf = PdfReader(BytesIO(raw))
        if pdf.is_encrypted:
            raise ValueError("Upload an unlocked copy of the course PDF.")
        if len(pdf.pages) > 200:
            raise ValueError("Choose a PDF of up to 200 pages.")
        rows, skipped = [], 0
        for page, sheet in enumerate(pdf.pages, start=1):
            words = (sheet.extract_text() or "").split()
            if not words:
                skipped += 1
                continue
            start = 0
            while start < len(words):
                end = min(start + 160, len(words))
                rows.append(dict(source=source, id=f"{source}-p{page}-{start}", title=path.name,
                                 heading=f"PDF page {page}", page=page, url="", text=" ".join(words[start:end])))
                if len(rows) > MAX_PASSAGES:
                    raise ValueError("This PDF is too long. Upload a shorter course reading.")
                if end == len(words):
                    break
                start = end - 24
    except (PyPdfError, OSError, KeyError, IndexError, RecursionError) as exc:
        raise ValueError("Could not read this PDF. Try exporting a new copy.") from exc
    if not rows:
        raise ValueError("No selectable text was found. Upload a searchable PDF or apply OCR first.")
    return rows, skipped


def add_pdf(library, course, file, base):
    course = (course or "").strip()
    if not course or len(course) > 80 or course.casefold() == BUILTIN.casefold():
        raise ValueError("Enter a course name of up to 80 characters for the uploaded materials.")
    course = next((name for name in library if name.casefold() == course.casefold()), course)
    if not file:
        raise ValueError("Choose a course PDF first.")
    path = Path(file)
    if path.suffix.lower() != ".pdf" or path.stat().st_size > 20_000_000:
        raise ValueError("Choose a PDF no larger than 20 MB.")
    raw = path.read_bytes()
    source = "pdf-" + digest(raw)[:20]
    previous = library.get(course)
    if previous and any(row["source"] == source for row in previous.rows):
        return library, course, f"This PDF is already indexed for {course}."
    added, skipped = extract(path, raw, source)
    if len(added) + sum(len(idx.rows) for idx in library.values()) > MAX_PASSAGES:
        raise ValueError("This session can hold up to 240 passages. Use a shorter reading.")
    rows = (previous.rows if previous else []) + added
    key = os.getenv("GEMINI_API_KEY")
    semantic = previous.vectors is not None if previous else bool(key and base.vectors)
    vectors = None
    mode = "lexical"
    if semantic:
        docs = dict(previous.vectors["docs"]) if previous else {}
        try:
            for row in added:
                docs[row["id"]] = vector(f"title: {row['title']} / {row['heading']} | text: {row['text']}", key)
        except Exception as exc:
            raise ValueError("Gemini could not embed the PDF. Your existing course materials are intact. Try again.") from exc
        vectors = {"docs": docs, "queries": base.vectors["queries"]}
        mode = MODEL
    fingerprint = digest("|".join(r["id"] for r in rows).encode())[:16]
    index = Index.from_passages(rows, vectors, f"course-pdf-v1:{fingerprint}:{mode}", multipart=base.multipart)
    pages = len({row["page"] for row in added})
    status = f"{path.name}: {pages} text page(s), {len(added)} new passage(s). Retrieval: {mode}."
    if skipped:
        status += f" Skipped {skipped} page(s) without selectable text."
    return {**library, course: index}, course, status
