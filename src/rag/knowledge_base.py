"""
Builds and persists a Chroma collection from the markdown protocol docs
in data/protocols/. Run once at deploy time (or on startup if the
collection is empty) — see api/main.py lifespan handler.
"""
from __future__ import annotations

import logging
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from config.settings import get_settings

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "emergency_protocols"


def _chunk_markdown(text: str, max_chars: int = 800) -> list[str]:
    """Naive section-aware chunking: split on '## ' headers, then hard-cap length."""
    sections = text.split("\n## ")
    chunks: list[str] = []
    for i, section in enumerate(sections):
        section = section if i == 0 else "## " + section
        section = section.strip()
        if not section:
            continue
        if len(section) <= max_chars:
            chunks.append(section)
        else:
            for start in range(0, len(section), max_chars):
                chunks.append(section[start : start + max_chars])
    return chunks


def build_or_load_collection(force_rebuild: bool = False) -> chromadb.Collection:
    settings = get_settings()
    settings.vector_store_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(settings.vector_store_dir))
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=settings.embedding_model
    )

    existing = [c.name for c in client.list_collections()]
    if _COLLECTION_NAME in existing and not force_rebuild:
        return client.get_collection(_COLLECTION_NAME, embedding_function=embedding_fn)

    if _COLLECTION_NAME in existing:
        client.delete_collection(_COLLECTION_NAME)

    collection = client.create_collection(_COLLECTION_NAME, embedding_function=embedding_fn)

    ids, documents, metadatas = [], [], []
    for path in sorted(settings.protocols_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for idx, chunk in enumerate(_chunk_markdown(text)):
            ids.append(f"{path.stem}::{idx}")
            documents.append(chunk)
            metadatas.append({"source": path.name})

    if documents:
        collection.add(ids=ids, documents=documents, metadatas=metadatas)
        logger.info("Indexed %d chunks from %s", len(documents), settings.protocols_dir)
    else:
        logger.warning("No protocol documents found in %s", settings.protocols_dir)

    return collection
