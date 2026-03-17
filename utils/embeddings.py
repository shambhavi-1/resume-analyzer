"""
utils/embeddings.py
────────────────────
Sentence-Transformer embedding pipeline.
Model: all-MiniLM-L6-v2 (fast, accurate, 384-dim)

Handles:
 - Text encoding (single string or batch)
 - FAISS index creation & querying
 - Embedding caching (in-memory LRU)
"""

import os
import hashlib
import pickle
from pathlib import Path
from typing import Optional, Union
from functools import lru_cache

import numpy as np
from loguru import logger

# ── Lazy imports (heavy deps) ────────────────────────────────────────────────
_model = None
_faiss = None

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_DIR = Path(__file__).parent.parent / "models"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output dimension


def _get_model():
    """Lazy-load Sentence Transformer model."""
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {MODEL_NAME}")
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(MODEL_NAME, cache_folder=str(CACHE_DIR))
            logger.success(f"Model loaded. Embedding dim: {_model.get_sentence_embedding_dimension()}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    return _model


def _get_faiss():
    """Lazy-load FAISS library."""
    global _faiss
    if _faiss is None:
        import faiss as _faiss_module
        _faiss = _faiss_module
    return _faiss


# ── Text chunking ────────────────────────────────────────────────────────────

def _chunk_text(text: str, max_tokens: int = 256, overlap: int = 32) -> list[str]:
    """
    Split long text into overlapping word-level chunks.
    Sentence Transformers have a max sequence length (~256 tokens).
    """
    words = text.split()
    if len(words) <= max_tokens:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_tokens, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += max_tokens - overlap

    return chunks


# ── LRU embedding cache ──────────────────────────────────────────────────────
_embed_cache: dict[str, np.ndarray] = {}


def _text_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def encode_text(text: str, use_cache: bool = True) -> np.ndarray:
    """
    Encode a single text string into a normalized embedding vector.
    
    For long texts, encodes in chunks and returns the mean-pooled vector.
    Returns a float32 numpy array of shape (EMBEDDING_DIM,).
    """
    if not text or not text.strip():
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    text = text.strip()
    cache_key = _text_hash(text)

    if use_cache and cache_key in _embed_cache:
        return _embed_cache[cache_key]

    model = _get_model()
    chunks = _chunk_text(text)

    if len(chunks) == 1:
        embedding = model.encode(
            chunks[0],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    else:
        # Encode all chunks and mean-pool
        chunk_embeddings = model.encode(
            chunks,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
            batch_size=16,
        )
        embedding = chunk_embeddings.mean(axis=0)
        # Re-normalize after mean-pooling
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

    embedding = embedding.astype(np.float32)

    if use_cache:
        _embed_cache[cache_key] = embedding

    return embedding


def encode_batch(texts: list[str], use_cache: bool = True) -> np.ndarray:
    """
    Encode a list of texts. Returns float32 array of shape (N, EMBEDDING_DIM).
    """
    embeddings = np.zeros((len(texts), EMBEDDING_DIM), dtype=np.float32)
    for i, text in enumerate(texts):
        embeddings[i] = encode_text(text, use_cache=use_cache)
    return embeddings


# ── FAISS Index ──────────────────────────────────────────────────────────────

class ResumeIndex:
    """
    FAISS-backed vector index for resume embeddings.
    Supports add, search, save, and load operations.
    """

    def __init__(self, dim: int = EMBEDDING_DIM, use_gpu: bool = False):
        faiss = _get_faiss()
        self.dim = dim
        self.use_gpu = use_gpu

        # Inner product index (works with L2-normalized vectors = cosine sim)
        self.index = faiss.IndexFlatIP(dim)

        if use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                logger.info("FAISS index running on GPU")
            except Exception:
                logger.warning("GPU not available; using CPU FAISS index.")

        # Metadata store (id → info dict)
        self._metadata: list[dict] = []
        logger.debug(f"FAISS index initialized (dim={dim})")

    def add(self, text: str, metadata: Optional[dict] = None) -> int:
        """
        Encode text and add to the index.
        Returns the integer ID assigned to this entry.
        """
        embedding = encode_text(text)
        embedding_2d = embedding.reshape(1, -1)
        self.index.add(embedding_2d)
        self._metadata.append(metadata or {"text": text[:200]})
        idx = len(self._metadata) - 1
        logger.debug(f"Added entry {idx} to FAISS index")
        return idx

    def add_batch(self, texts: list[str], metadatas: Optional[list[dict]] = None) -> list[int]:
        """Add multiple texts at once. Returns list of assigned IDs."""
        if not texts:
            return []
        embeddings = encode_batch(texts)
        self.index.add(embeddings)
        ids = []
        for i, text in enumerate(texts):
            meta = (metadatas[i] if metadatas else None) or {"text": text[:200]}
            self._metadata.append(meta)
            ids.append(len(self._metadata) - 1)
        return ids

    def search(self, query_text: str, k: int = 5) -> list[dict]:
        """
        Find the k most similar entries to the query text.
        Returns list of dicts with keys: id, score, metadata.
        """
        if self.index.ntotal == 0:
            return []

        k = min(k, self.index.ntotal)
        query_emb = encode_text(query_text).reshape(1, -1)
        scores, indices = self.index.search(query_emb, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append({
                "id": int(idx),
                "score": float(score),
                "metadata": self._metadata[idx],
            })
        return results

    def query_similarity(self, text_a: str, text_b: str) -> float:
        """
        Compute cosine similarity between two texts.
        Returns a float in [-1, 1] (for L2-normalized vectors: [0, 1]).
        """
        emb_a = encode_text(text_a)
        emb_b = encode_text(text_b)
        similarity = float(np.dot(emb_a, emb_b))
        return similarity

    def save(self, path: Union[str, Path]) -> None:
        """Save index and metadata to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        faiss = _get_faiss()
        # Move back to CPU before saving if on GPU
        index_cpu = faiss.index_gpu_to_cpu(self.index) if self.use_gpu else self.index
        faiss.write_index(index_cpu, str(path / "index.faiss"))
        with open(path / "metadata.pkl", "wb") as f:
            pickle.dump(self._metadata, f)
        logger.info(f"FAISS index saved to {path}")

    def load(self, path: Union[str, Path]) -> None:
        """Load index and metadata from disk."""
        path = Path(path)
        faiss = _get_faiss()
        self.index = faiss.read_index(str(path / "index.faiss"))
        with open(path / "metadata.pkl", "rb") as f:
            self._metadata = pickle.load(f)
        logger.info(f"FAISS index loaded from {path} ({self.index.ntotal} entries)")

    @property
    def size(self) -> int:
        return self.index.ntotal


# ── Convenience singleton ────────────────────────────────────────────────────
_global_index: Optional[ResumeIndex] = None


def get_global_index() -> ResumeIndex:
    """Return the global singleton ResumeIndex (created on first call)."""
    global _global_index
    if _global_index is None:
        _global_index = ResumeIndex()
    return _global_index
