"""Metadata-filtered vector retrieval and cross-encoder re-ranking."""

import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder, SentenceTransformer


class VectorStore:
    """FAISS index plus its row-aligned chunk metadata."""

    def __init__(self, directory: Path):
        self.index = faiss.read_index(str(directory / "index.faiss"))

        parquet, csv = directory / "chunk_metadata.parquet", directory / "chunk_metadata.csv"
        if parquet.exists():
            self.metadata = pd.read_parquet(parquet)
        elif csv.exists():
            self.metadata = pd.read_csv(csv)
        else:
            raise FileNotFoundError(f"No chunk metadata found in {directory}. Run: python -m src.ingest")

        if self.index.ntotal != len(self.metadata):
            raise RuntimeError("FAISS index and chunk metadata are out of sync. Re-run ingestion.")

    def brands(self) -> list[str]:
        return sorted(self.metadata["brand"].unique())

    def models(self, brand: str) -> list[str]:
        return sorted(self.metadata.loc[self.metadata["brand"] == brand, "model"].unique())

    def rows_for(self, brand: str, model: str) -> np.ndarray:
        """Row positions (== FAISS ids) of the selected vehicle's chunks."""
        mask = (self.metadata["brand"] == brand) & (self.metadata["model"] == model)
        return np.flatnonzero(mask.to_numpy())


class Retriever:
    def __init__(self, cfg: dict, store: VectorStore):
        self.store = store
        self.query_instruction = cfg["models"]["bge_query_instruction"]
        self.embedder = SentenceTransformer(cfg["models"]["embedding"])
        self.reranker = CrossEncoder(cfg["models"]["reranker"])

    def _embed_query(self, query: str) -> np.ndarray:
        vec = self.embedder.encode([self.query_instruction + query], convert_to_numpy=True)
        vec = vec.astype("float32")
        faiss.normalize_L2(vec)
        return vec[0]

    def retrieve(self, query: str, brand: str, model: str, top_k: int) -> tuple[pd.DataFrame, float]:
        """Filter to the selected vehicle FIRST, then rank only its chunks by cosine similarity."""
        rows = self.store.rows_for(brand, model)
        if rows.size == 0:
            return pd.DataFrame(), 0.0

        start = time.perf_counter()
        query_vec = self._embed_query(query)
        vectors = np.vstack([self.store.index.reconstruct(int(r)) for r in rows])
        scores = vectors @ query_vec
        best = np.argsort(-scores)[:top_k]
        elapsed = time.perf_counter() - start

        result = self.store.metadata.iloc[rows[best]].copy()
        result["similarity_score"] = scores[best]
        return result.reset_index(drop=True), elapsed

    def rerank(self, query: str, candidates: pd.DataFrame, top_n: int) -> tuple[pd.DataFrame, float]:
        """Re-score candidates with the cross-encoder and keep the best top_n."""
        if candidates.empty:
            return candidates, 0.0

        start = time.perf_counter()
        scores = self.reranker.predict([(query, text) for text in candidates["text"]])
        elapsed = time.perf_counter() - start

        result = candidates.assign(rerank_score=scores)
        result = result.sort_values("rerank_score", ascending=False).head(top_n)
        return result.reset_index(drop=True), elapsed
