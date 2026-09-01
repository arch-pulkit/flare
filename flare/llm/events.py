# BOUNDARY 3: flare/llm/events.py → imports ONLY from flare.detection.events and stdlib
# LLM layer knows about anomalies, not ASTR-O internals
from __future__ import annotations

from dataclasses import dataclass


#one toke, and its log probability(value object)
@dataclass(frozen=True)
class LogprobToken:
    token: str
    logprob: float


#one chunk from retrieval(value object)
@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    source: str        # filename only — must match reference_registry.json exactly
    text: str
    rank: int
    dense_score: float
    sparse_score: float
    rrf_score: float
    source_tier: str   # "CRITICAL" | "SUPPORTING" | "REFERENCE"


#the llm answer event (domain event)
@dataclass(frozen=True)
class RecommendationGenerated:
    event_id: str
    incident_id: str
    parent_event_id: str         # AnomalyDetected.event_id
    mission_id: str
    llm_text: str
    logprobs: tuple[LogprobToken, ...]
    retrieved_chunks: tuple[RetrievedChunk, ...]
    query: str
    retrieval_method: str        # "hybrid"
    top_k: int
    all_ranked_chunks: tuple[dict, ...]  # {chunk_id, rank, dense_score, sparse_score, rrf_score}
    llm_backend: str             # "openai:gpt-4o-mini" | "llama_cpp:Llama-3.1-8B-Q4"
    generated_at: float
