import hashlib
import io
import re
import numpy as np
from pypdf import PdfReader
from pypdf.errors import PdfReadError

MAX_PDF_BYTES = 20 * 1024 * 1024
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
DEFAULT_SIMILARITY_THRESHOLD = 0.25


class PDFProcessingError(ValueError):
    pass


def process_pdf(pdf_bytes: bytes, filename: str) -> dict:
    if not pdf_bytes:
        raise PDFProcessingError("The uploaded PDF is empty.")
    if len(pdf_bytes) > MAX_PDF_BYTES:
        raise PDFProcessingError("PDF is too large. Maximum allowed size is 20 MB.")
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes), strict=False)
        if reader.is_encrypted:
            try:
                result = reader.decrypt("")
                if result == 0:
                    raise PDFProcessingError(
                        "This PDF is encrypted/password-protected. Upload an unlocked PDF."
                    )
            except PDFProcessingError:
                raise
            except Exception as exc:
                raise PDFProcessingError("This PDF is encrypted/password-protected.") from exc
        page_count = len(reader.pages)
        if page_count == 0:
            raise PDFProcessingError("The PDF has no pages.")
        chunks = []
        for page_index, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = re.sub(r"\s+", " ", text).strip()
            if not text:
                continue
            words = text.split()
            start = 0
            while start < len(words):
                part = " ".join(words[start : start + CHUNK_SIZE])
                if part:
                    chunk_id = f"p{page_index}_c{len(chunks) + 1}"
                    chunks.append(
                        {
                            "chunk_id": chunk_id,
                            "page_number": page_index,
                            "source_filename": filename,
                            "text": part,
                        }
                    )
                if start + CHUNK_SIZE >= len(words):
                    break
                start += CHUNK_SIZE - CHUNK_OVERLAP
        if not chunks:
            raise PDFProcessingError(
                "No selectable text was found. This appears to be an image-only/scanned PDF; run OCR first."
            )
        fingerprint = hashlib.sha256(pdf_bytes).hexdigest()
        return {
            "filename": filename,
            "page_count": page_count,
            "chunks": chunks,
            "fingerprint": fingerprint,
        }
    except PDFProcessingError:
        raise
    except (PdfReadError, Exception) as exc:
        raise PDFProcessingError(
            f"Could not read this PDF. It may be corrupted or unsupported: {exc}"
        ) from exc


def embed_chunks(chunks, model):
    texts = [c["text"] for c in chunks]
    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype=np.float32)


def semantic_search(
    query: str,
    chunks: list,
    embeddings,
    model=None,
    top_k: int = 3,
    min_similarity: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> list:
    """Retrieve up to top_k chunks whose cosine similarity is at least min_similarity."""
    if not query or not query.strip() or not chunks or embeddings is None:
        return []
    if model is None:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    query_vector = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]
    scores = np.asarray(embeddings) @ np.asarray(query_vector)
    indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for index in indices:
        score = float(scores[int(index)])
        if score < min_similarity:
            continue
        item = dict(chunks[int(index)])
        item["score"] = score
        results.append(item)
    return results


def clear_document_state(session_state):
    session_state.pdf_doc = None
    session_state.pdf_embeddings = None
    session_state.pdf_fingerprint = None
