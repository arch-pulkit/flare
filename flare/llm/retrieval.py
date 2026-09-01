# BOUNDARY: flare/llm/retrieval.py → imports from flare.llm.events, flare.detection.events,
#           and stdlib/third-party only. No flare.state, flare.audit, or flare.pipeline.
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from flare.detection.events import AnomalyDetected
from flare.llm.events import RetrievedChunk

logger = logging.getLogger(__name__)


#hybrid retriever
class KnowledgeBase:
    def __init__(self, docs_folder: Path, registry_config_path: Path) -> None:
        config: dict[str, Any] = json.loads(registry_config_path.read_text(encoding="utf-8"))

        # Build filename → {doc_id, source_tier} lookup from registry config
        self._file_meta: dict[str, dict[str, str]] = {
            doc["filename"]: {
                "doc_id":      doc["document_id"],
                "source_tier": doc["source_tier"],
            }
            for doc in config["documents"]
        }

        # Load, chunk, and index documents
        self._chunks: list[dict[str, str]] = []
        for filename, meta in self._file_meta.items():
            filepath = docs_folder / filename
            if not filepath.exists():
                logger.warning("Knowledge base document not found: %s", filepath)
                continue
            text = filepath.read_text(encoding="utf-8")
            paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) >= 200]
            for n, para in enumerate(paragraphs):
                self._chunks.append({
                    "chunk_id":   f"{meta['doc_id']}_chunk_{n:03d}",
                    "text":       para,
                    "source":     filename,   # filename only — never a path
                    "source_tier": meta["source_tier"],
                })
        logger.info("KnowledgeBase: loaded %d chunks from %d documents", len(self._chunks), len(self._file_meta))

        chunk_texts = [c["text"] for c in self._chunks]

        # Sparse index (TF-IDF)
        self._tfidf = TfidfVectorizer()
        self._tfidf_matrix = self._tfidf.fit_transform(chunk_texts)

        # Dense index (sentence-transformers)
        from sentence_transformers import SentenceTransformer  # lazy import — large dep
        self._encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self._dense_matrix: np.ndarray = self._encoder.encode(
            chunk_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        max_per_source: int = 0,
    ) -> tuple[list[RetrievedChunk], list[dict[str, Any]]]:
        """Hybrid retrieval (TF-IDF + dense) fused with RRF.

        Args:
            max_per_source: when > 0, at most this many chunks per unique source
                document are included in the top_k results. Use 1 to prevent
                ASTR-O contradiction detection false positives from multi-state
                spec chunks within the same document.

        Returns:
            (top_k RetrievedChunk list, all_ranked_chunks list sorted by rrf_score desc)
        """
        n = len(self._chunks)

        # Sparse scores and ranks
        query_sparse = self._tfidf.transform([query])
        sparse_scores: np.ndarray = cosine_similarity(query_sparse, self._tfidf_matrix).flatten()
        sparse_order = np.argsort(-sparse_scores)
        sparse_rank: dict[int, int] = {int(idx): rank + 1 for rank, idx in enumerate(sparse_order)}

        # Dense scores and ranks
        query_dense: np.ndarray = self._encoder.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )
        dense_scores: np.ndarray = (query_dense @ self._dense_matrix.T).flatten()
        dense_order = np.argsort(-dense_scores)
        dense_rank: dict[int, int] = {int(idx): rank + 1 for rank, idx in enumerate(dense_order)}

        # RRF fusion: score = 1/(60+sparse_rank) + 1/(60+dense_rank)
        rrf_scores = np.array(
            [1.0 / (60 + sparse_rank[i]) + 1.0 / (60 + dense_rank[i]) for i in range(n)],
            dtype=np.float64,
        )
        rrf_order = np.argsort(-rrf_scores)

        # Build all_ranked_chunks (complete, not truncated)
        all_ranked: list[dict[str, Any]] = [
            {
                "chunk_id":    self._chunks[int(idx)]["chunk_id"],
                "rank":        rank + 1,
                "dense_score": float(dense_scores[int(idx)]),
                "sparse_score": float(sparse_scores[int(idx)]),
                "rrf_score":   float(rrf_scores[int(idx)]),
            }
            for rank, idx in enumerate(rrf_order)
        ]

        # Build top_k RetrievedChunk list (with optional source diversity)
        top_chunks: list[RetrievedChunk] = []
        source_counts: dict[str, int] = {}
        for idx in rrf_order:
            if len(top_chunks) >= top_k:
                break
            chunk = self._chunks[int(idx)]
            if max_per_source > 0:
                count = source_counts.get(chunk["source"], 0)
                if count >= max_per_source:
                    continue
                source_counts[chunk["source"]] = count + 1
            top_chunks.append(
                RetrievedChunk(
                    chunk_id=chunk["chunk_id"],
                    source=chunk["source"],
                    text=chunk["text"],
                    rank=len(top_chunks) + 1,
                    dense_score=float(dense_scores[int(idx)]),
                    sparse_score=float(sparse_scores[int(idx)]),
                    rrf_score=float(rrf_scores[int(idx)]),
                    source_tier=chunk["source_tier"],
                )
            )

        return top_chunks, all_ranked
