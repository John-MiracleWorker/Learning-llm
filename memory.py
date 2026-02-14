"""
Continual Learning LLM — Short-Term Episodic Memory (ChromaDB)

Provides the "Wake Phase" memory layer:
  • Store new facts taught by the user
  • Semantic retrieval (RAG) for injecting context into prompts
  • Admin operations: list, delete, mark-as-learned, wipe
"""

import time
import uuid
from typing import Optional

import chromadb
from chromadb.config import Settings

import config


# ── Singleton Client ───────────────────────────────────────────────────────

_client: Optional[chromadb.ClientAPI] = None


def _get_client() -> chromadb.ClientAPI:
    """Lazy-init a persistent ChromaDB client (local, no Docker)."""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=config.CHROMADB_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
    return _client


def get_collection() -> chromadb.Collection:
    """Return (or create) the short-term memory collection."""
    client = _get_client()
    return client.get_or_create_collection(
        name=config.CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# ── Write Operations ──────────────────────────────────────────────────────


def store_fact(fact_text: str, source: str = "user") -> str:
    """
    Store a single fact in short-term memory.

    Args:
        fact_text: The raw fact string to remember.
        source:    Origin tag — "user" for manually taught, "correction" for
                   corrections, etc.

    Returns:
        The unique ID assigned to this fact.
    """
    collection = get_collection()
    fact_id = f"fact-{uuid.uuid4().hex[:12]}"
    timestamp = time.time()

    collection.add(
        ids=[fact_id],
        documents=[fact_text],
        metadatas=[{
            "source": source,
            "timestamp": timestamp,
            "learned": "false",  # flipped to "true" after Sleep cycle
        }],
    )
    return fact_id


# ── Read Operations ───────────────────────────────────────────────────────


def retrieve(query: str, top_k: int = config.RAG_TOP_K) -> list[dict]:
    """
    Semantic search over short-term memory.

    Returns a list of dicts with keys: id, text, distance, metadata.
    Results are filtered by RAG_RELEVANCE_THRESHOLD (cosine distance;
    lower distance = higher similarity).
    """
    collection = get_collection()

    if collection.count() == 0:
        return []

    # Clamp top_k to available documents
    effective_k = min(top_k, collection.count())

    results = collection.query(
        query_texts=[query],
        n_results=effective_k,
    )

    hits = []
    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]
        # ChromaDB cosine distance: 0 = identical, 2 = opposite
        if distance <= config.RAG_RELEVANCE_THRESHOLD:
            hits.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "distance": distance,
                "metadata": results["metadatas"][0][i],
            })
    return hits


def get_all_facts(only_unlearned: bool = False) -> list[dict]:
    """
    Return every fact in the collection.

    Args:
        only_unlearned: If True, filter to facts not yet baked into weights.
    """
    collection = get_collection()

    if collection.count() == 0:
        return []

    if only_unlearned:
        results = collection.get(
            where={"learned": "false"},
        )
    else:
        results = collection.get()

    facts = []
    for i in range(len(results["ids"])):
        facts.append({
            "id": results["ids"][i],
            "text": results["documents"][i],
            "metadata": results["metadatas"][i],
        })
    return facts


def get_unlearned_facts() -> list[dict]:
    """Convenience wrapper: return only facts pending Sleep cycle."""
    return get_all_facts(only_unlearned=True)


# ── Delete / Update Operations ────────────────────────────────────────────


def delete_fact(fact_id: str) -> None:
    """Remove a single fact by ID (admin curation before Sleep)."""
    collection = get_collection()
    collection.delete(ids=[fact_id])


def mark_as_learned(fact_ids: list[str]) -> None:
    """
    After a successful Sleep cycle, mark these facts as baked into weights.
    They remain in ChromaDB for RAG but won't be re-trained.
    """
    collection = get_collection()
    for fid in fact_ids:
        collection.update(
            ids=[fid],
            metadatas=[{"learned": "true"}],
        )


def wipe_learned() -> int:
    """
    Remove all facts that have been successfully learned (post-Sleep cleanup).
    Returns the count of deleted facts.
    """
    collection = get_collection()
    learned = collection.get(where={"learned": "true"})
    count = len(learned["ids"])
    if count > 0:
        collection.delete(ids=learned["ids"])
    return count


def count() -> int:
    """Total number of facts in short-term memory."""
    return get_collection().count()


def count_unlearned() -> int:
    """Number of facts not yet baked into weights."""
    return len(get_unlearned_facts())


# ── RAG Prompt Injection ──────────────────────────────────────────────────


def build_rag_context(query: str) -> tuple[str, bool]:
    """
    Build a context string for system-prompt injection.

    Args:
        query: The user's latest chat message.

    Returns:
        (context_string, has_rag) — the formatted context block and a
        boolean indicating whether any facts were retrieved.
    """
    hits = retrieve(query)
    if not hits:
        return "", False

    lines = ["[Short-Term Memory — recently learned facts]"]
    for i, hit in enumerate(hits, 1):
        lines.append(f"{i}. {hit['text']}")
    lines.append("[End of Short-Term Memory]\n")

    return "\n".join(lines), True
