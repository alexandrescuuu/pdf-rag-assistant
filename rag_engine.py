import io
import re
from dataclasses import dataclass
import numpy as np
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except Exception:
    HAS_SBERT = False


@dataclass
class DocumentChunk:
    chunk_id: str
    source_file: str
    page_num: int
    text: str


class PDFRagEngine:
    """Multi-PDF Retrieval-Augmented Generation engine with exact page-level citations."""

    def __init__(self, chunk_size: int = 550, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks: list[DocumentChunk] = []
        self.tfidf = TfidfVectorizer()
        self.tfidf_matrix = None
        self.dense_embeddings = None
        self.embedder = None

        if HAS_SBERT:
            try:
                self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:
                self.embedder = None

    def _chunk_text(self, text: str) -> list[str]:
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            return []
        pieces = []
        start = 0
        step = max(50, self.chunk_size - self.chunk_overlap)
        while start < len(cleaned):
            pieces.append(cleaned[start : start + self.chunk_size])
            start += step
        return pieces

    def index_pdfs(self, pdf_files: list[tuple[str, bytes]]) -> int:
        """Parses multiple (filename, raw_bytes) PDFs and builds lexical + dense vector indices."""
        self.chunks = []
        for fname, raw_bytes in pdf_files:
            reader = PdfReader(io.BytesIO(raw_bytes))
            for page_idx, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                for c_idx, chunk_str in enumerate(self._chunk_text(page_text), start=1):
                    self.chunks.append(
                        DocumentChunk(
                            chunk_id=f"{fname}:p{page_idx}:c{c_idx}",
                            source_file=fname,
                            page_num=page_idx,
                            text=chunk_str,
                        )
                    )

        if not self.chunks:
            return 0

        corpus = [c.text for c in self.chunks]
        self.tfidf_matrix = self.tfidf.fit_transform(corpus)

        if self.embedder is not None:
            self.dense_embeddings = self.embedder.encode(corpus, normalize_embeddings=True)
        return len(self.chunks)

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        if not self.chunks or self.tfidf_matrix is None:
            return []

        q_tfidf = self.tfidf.transform((query,))
        sparse_scores = cosine_similarity(q_tfidf, self.tfidf_matrix).ravel()

        if self.embedder is not None and self.dense_embeddings is not None:
            q_vec = self.embedder.encode((query,), normalize_embeddings=True)
            dense_scores = np.dot(self.dense_embeddings, q_vec.T).ravel()
            combined = 0.65 * np.clip(dense_scores, 0, 1) + 0.35 * sparse_scores
        else:
            combined = sparse_scores

        ranked_indices = np.argsort(combined)[::-1][:top_k]
        results = []
        for idx in ranked_indices:
            i = int(idx)
            chunk = self.chunks[i]
            results.append({
                "Source": chunk.source_file,
                "Page": chunk.page_num,
                "Relevance Score": round(float(combined.item(i)), 4),
                "Snippet": chunk.text,
            })
        return results

    def generate_grounded_answer(self, query: str, top_k: int = 3, min_score: float = 0.12) -> tuple[str, list[dict]]:
        """Synthesizes a grounded answer with inline page citations or abstains if evidence is weak."""
        hits = self.retrieve(query, top_k=top_k)
        first_hit = next(iter(hits), None)
        if first_hit is None or first_hit["Relevance Score"] < min_score:
            return (
                "I could not find reliable evidence in the uploaded PDFs to answer this question.",
                hits,
            )

        q_terms = set(re.findall(r"[a-zA-ZÀ-ÿ0-9]+", query.lower())) - {
            "what", "how", "why", "the", "is", "are", "of", "in", "to", "a",
            "comment", "quoi", "est", "les", "des", "une", "pour", "avec", "dans", "sur",
        }
        cited_sentences = []
        for h in hits:
            if h["Relevance Score"] < min_score:
                continue
            sentences = re.split(r"(?<=[.!?])\s+", h["Snippet"])
            best_sent = max(
                sentences,
                key=lambda s: len(q_terms & set(re.findall(r"[a-zA-ZÀ-ÿ0-9]+", s.lower()))),
            )
            cited_sentences.append(f"- {best_sent.strip()} **[{h['Source']}, p. {h['Page']}]**")

        answer = "Based on the indexed PDF documents:\n\n" + "\n".join(cited_sentences)
        return answer, hits
