# BOUNDARY: flare/llm/handlers.py → imports from flare.llm.*, flare.detection.events, and stdlib only
from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Callable

from flare.detection.events import AnomalyDetected
from flare.llm.backend import LLMBackend
from flare.llm.events import LogprobToken, RecommendationGenerated
from flare.llm.prompt import SYSTEM_PROMPT, build_query, build_user_prompt, parse_llm_response
from flare.llm.retrieval import KnowledgeBase

logger = logging.getLogger(__name__)


def make_llm_handler(
    knowledge_base: KnowledgeBase,
    llm_backend: LLMBackend,
    mission_id: str,
    top_k: int = 5,
    max_per_source: int = 1,
) -> Callable[[AnomalyDetected], RecommendationGenerated]:
    def llm_handler(anomaly: AnomalyDetected) -> RecommendationGenerated:
        query = build_query(anomaly)
        chunks, all_ranked = knowledge_base.retrieve(query, top_k=top_k, max_per_source=max_per_source)

        user_prompt = build_user_prompt(anomaly, chunks)
        try:
            response = llm_backend.generate(system=SYSTEM_PROMPT, user=user_prompt)
            parsed = parse_llm_response(response["text"])
            llm_text = json.dumps(parsed)
            logprobs: tuple[LogprobToken, ...] = tuple(
                LogprobToken(token=lp["token"], logprob=lp["logprob"])
                for lp in response["logprobs"]
            )
        except Exception as exc:
            logger.error(
                "LLM backend failed for incident %s: %s",
                anomaly.incident_id, exc,
            )
            llm_text = json.dumps({
                "ASSESSMENT": "LLM_UNAVAILABLE",
                "LIKELY_CAUSE": "NOT PROVIDED",
                "RECOMMENDED_ACTION": "NOT PROVIDED",
                "URGENCY": "NOT PROVIDED",
            })
            logprobs = ()

        return RecommendationGenerated(
            event_id=str(uuid.uuid4()),
            incident_id=anomaly.incident_id,
            parent_event_id=anomaly.event_id,
            mission_id=mission_id,
            llm_text=llm_text,
            logprobs=logprobs,
            retrieved_chunks=tuple(chunks),
            query=query,
            retrieval_method="hybrid",
            top_k=top_k,
            all_ranked_chunks=tuple(all_ranked),
            llm_backend=llm_backend.backend_id,
            generated_at=time.time(),
        )

    return llm_handler
