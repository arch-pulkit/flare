# BOUNDARY: flare/audit/span_builder.py → imports from flare.llm.events and stdlib only
from __future__ import annotations

from flare.llm.events import RecommendationGenerated


def build_span(event: RecommendationGenerated) -> dict:
    """Pure function — no I/O, no side effects. Same input always produces same output."""
    return {
        "span_id": event.event_id,
        "trace_id": event.incident_id,
        "retrieved_chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "text": chunk.text,
                "metadata": {"source_tier": chunk.source_tier},
            }
            for chunk in event.retrieved_chunks
        ],
        "retrieval_metadata": {
            "query": event.query,
            "retrieval_method": event.retrieval_method,
            "top_k": event.top_k,
            # flat chunk_id list — separate from top-level retrieved_chunks objects
            # required by ASTR-O source_verifier; missing this → PARTIAL_EVALUATION
            "retrieved_chunks": [chunk.chunk_id for chunk in event.retrieved_chunks],
            "all_ranked_chunks": list(event.all_ranked_chunks),
        },
        "llm_response": {
            "text": event.llm_text,
            "logprobs": [
                {"token": t.token, "logprob": t.logprob}
                for t in event.logprobs
            ],
        },
    }
