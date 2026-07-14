"""
Thin retrieval interface. Agents call `retrieve_protocol()` — they don't
touch Chroma directly, so the backend (Chroma vs FAISS) can be swapped
by only editing this file + knowledge_base.py.
"""
from __future__ import annotations

from functools import lru_cache

from config.settings import get_settings
from core.constants import EMERGENCY_PROTOCOL_MAP
from core.schemas import EmergencyType
from rag.knowledge_base import build_or_load_collection


@lru_cache
def _collection():
    return build_or_load_collection()


def retrieve_protocol(query: str, emergency_type: EmergencyType | None = None) -> list[str]:
    """
    Semantic search over the protocol docs. If `emergency_type` is given,
    biases retrieval by prepending the mapped protocol filename as a
    soft filter hint in the query — Chroma has no hard filter here
    because a High-risk crowd situation can still benefit from
    cross-referencing e.g. the medical protocol.
    """
    settings = get_settings()
    coll = _collection()

    search_query = query
    if emergency_type is not None:
        hint_file = EMERGENCY_PROTOCOL_MAP.get(emergency_type, "")
        search_query = f"[{hint_file}] {query}"

    results = coll.query(query_texts=[search_query], n_results=settings.rag_top_k)
    documents = results.get("documents", [[]])[0]
    return documents
