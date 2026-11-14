import io
import unittest
import numpy as np
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
from src.pdf_rag import (
    PDFProcessingError,
    clear_document_state,
    embed_chunks,
    process_pdf,
    semantic_search,
)


def make_pdf(text: str = "This is a sample educational PDF with searchable text.") -> bytes:
    """Create a minimal valid single-page PDF with extractable text using pypdf primitives."""
    writer = PdfWriter()
    page = writer.add_blank_page(width=400, height=400)

    font_dict = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_ref = writer._add_object(font_dict)
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref})}
    )

    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = DecodedStreamObject()
    stream.set_data(f"BT /F1 12 Tf 20 350 Td ({escaped}) Tj ET".encode("latin-1"))
    stream_ref = writer._add_object(stream)
    page[NameObject("/Contents")] = stream_ref

    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


class FakeEmbeddingModel:
    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False):
        vectors = []
        for t in texts:
            lower = t.lower()
            v = np.array(
                [
                    1.0 if "python" in lower else 0.0,
                    1.0 if "neural" in lower or "deep" in lower else 0.0,
                    1.0 if "cooking" in lower else 0.0,
                    0.1,
                ],
                dtype=np.float32,
            )
            norm = np.linalg.norm(v)
            if normalize_embeddings and norm > 0:
                v = v / norm
            vectors.append(v)
        return np.asarray(vectors, dtype=np.float32)


class TestPDFRag(unittest.TestCase):
    def test_process_text_pdf(self):
        result = process_pdf(make_pdf(), "sample.pdf")
        self.assertEqual(result["page_count"], 1)
        self.assertTrue(result["chunks"])
        self.assertEqual(result["chunks"][0]["page_number"], 1)
        self.assertEqual(result["chunks"][0]["source_filename"], "sample.pdf")

    def test_empty_pdf_rejected(self):
        with self.assertRaises(PDFProcessingError):
            process_pdf(b"", "empty.pdf")

    def test_image_or_blank_pdf_rejected(self):
        writer = PdfWriter()
        writer.add_blank_page(width=300, height=300)
        buf = io.BytesIO()
        writer.write(buf)
        with self.assertRaises(PDFProcessingError):
            process_pdf(buf.getvalue(), "blank.pdf")

    def test_clear_document_state(self):
        class State:
            pdf_doc = {"x": 1}
            pdf_embeddings = [1]
            pdf_fingerprint = "abc"

        state = State()
        clear_document_state(state)
        self.assertIsNone(state.pdf_doc)
        self.assertIsNone(state.pdf_embeddings)
        self.assertIsNone(state.pdf_fingerprint)

    def test_semantic_search_and_similarity_threshold(self):
        chunks = [
            {"chunk_id": "p1_c1", "page_number": 1, "source_filename": "ai.pdf", "text": "Python programming language basics"},
            {"chunk_id": "p2_c2", "page_number": 2, "source_filename": "ai.pdf", "text": "Deep neural networks and backpropagation"},
        ]
        model = FakeEmbeddingModel()
        embeddings = embed_chunks(chunks, model)

        matched = semantic_search("Python programming", chunks, embeddings, model=model, top_k=3, min_similarity=0.5)
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0]["chunk_id"], "p1_c1")

        unrelated = semantic_search("Traditional cooking recipes", chunks, embeddings, model=model, top_k=3, min_similarity=0.5)
        self.assertEqual(unrelated, [])


if __name__ == "__main__":
    unittest.main()
