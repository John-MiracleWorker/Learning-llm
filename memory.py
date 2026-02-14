"""
Continual Learning LLM — Short-Term Episodic Memory (ChromaDB)

Provides the "Wake Phase" memory layer:
  • Store new facts taught by the user
  • Semantic retrieval (RAG) for injecting context into prompts
  • Admin operations: list, delete, mark-as-learned, wipe
  • LLM-as-a-Judge contradiction detection for fact supersession
"""

import re
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


# ── LLM-as-a-Judge Contradiction Detection (Bug 2B) ──────────────────────


def _llm_contradiction_check(new_fact: str, old_fact: str) -> bool:
    """
    Use the LLM to determine if a new fact logically contradicts an old fact.

    Dense embeddings measure semantic TOPICS, not logical DIRECTION.
    "I love Python" and "I hate Python" are nearly identical in embedding space.
    This two-stage verification uses the LLM as a judge to detect actual
    contradictions vs. merely related facts.

    Args:
        new_fact: The newly stated fact.
        old_fact: The existing fact to check against.

    Returns:
        True if the LLM judges them as contradictory (old should be superseded).
    """
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    adapter_file = __import__("os").path.join(config.ADAPTER_DIR, "adapters.safetensors")
    if __import__("os").path.exists(adapter_file):
        model, tokenizer = load(
            config.BASE_MODEL,
            adapter_path=config.ADAPTER_DIR,
            tokenizer_config={"trust_remote_code": True},
        )
    else:
        model, tokenizer = load(
            config.BASE_MODEL,
            tokenizer_config={"trust_remote_code": True},
        )

    messages = [
        {"role": "system", "content": (
            "You are a factual contradiction detector. "
            "Determine whether a NEW fact logically contradicts or updates an OLD fact. "
            "Related but non-contradictory facts should NOT be flagged. "
            "Answer ONLY 'YES' or 'NO'."
        )},
        {"role": "user", "content": (
            f'OLD FACT: "{old_fact}"\n'
            f'NEW FACT: "{new_fact}"\n\n'
            "Does the NEW FACT contradict or update the OLD FACT? (YES/NO)"
        )},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    raw = generate(
        model, tokenizer, prompt=prompt,
        max_tokens=32, sampler=make_sampler(temp=0.1),
    )
    # Strip thinking blocks
    raw = re.sub(r"<think>.*?</think>\s*", "", raw, flags=re.DOTALL).strip()

    del model, tokenizer
    import gc
    gc.collect()

    return "YES" in raw.upper()


# ── Write Operations ──────────────────────────────────────────────────────


def store_fact(fact_text: str, source: str = "user") -> tuple[str, list[dict]]:
    """
    Store a single fact in short-term memory with LLM-verified conflict detection.

    Two-stage RAG Verification (Bug 2B):
      Stage 1: ChromaDB retrieves candidates within SUPERSEDE_DISTANCE_THRESHOLD
      Stage 2: LLM judges whether each candidate is actually contradicted

    This prevents false supersession of related-but-distinct facts
    (e.g., "I love Python" vs "I love JavaScript" would NOT be superseded).

    Args:
        fact_text: The raw fact string to remember.
        source:    Origin tag — "user" for manually taught, "auto" for
                   auto-detected, "correction" for corrections.

    Returns:
        Tuple of (fact_id, list of superseded fact dicts).
    """
    collection = get_collection()
    fact_id = f"fact-{uuid.uuid4().hex[:12]}"
    timestamp = time.time()

    # Conflict detection: two-stage (embedding similarity + LLM judge)
    superseded = []
    if collection.count() > 0:
        effective_k = min(3, collection.count())
        results = collection.query(
            query_texts=[fact_text],
            n_results=effective_k,
        )
        for i in range(len(results["ids"][0])):
            distance = results["distances"][0][i]
            # Stage 1: Wider net for candidate retrieval (was 0.15, now 0.3)
            if distance < config.SUPERSEDE_DISTANCE_THRESHOLD:
                old_id = results["ids"][0][i]
                old_text = results["documents"][0][i]
                old_meta = results["metadatas"][0][i]

                # Skip already-superseded facts
                if old_meta.get("superseded") == "true":
                    continue

                # Stage 2: LLM-as-a-Judge — does it truly contradict?
                if _llm_contradiction_check(fact_text, old_text):
                    # Mark old fact as superseded
                    collection.update(
                        ids=[old_id],
                        metadatas=[{
                            **old_meta,
                            "superseded_by": fact_id,
                            "superseded": "true",
                        }],
                    )
                    superseded.append({
                        "id": old_id,
                        "text": old_text,
                        "metadata": old_meta,
                    })

    collection.add(
        ids=[fact_id],
        documents=[fact_text],
        metadatas=[{
            "source": source,
            "timestamp": timestamp,
            "learned": "false",  # flipped to "true" after Sleep cycle
        }],
    )
    return fact_id, superseded


# ── Read Operations ───────────────────────────────────────────────────────


def retrieve(query: str, top_k: int = config.RAG_TOP_K) -> list[dict]:
    """
    Semantic search over short-term memory.

    Returns a list of dicts with keys: id, text, distance, metadata.
    Results are filtered by RAG_RELEVANCE_THRESHOLD (cosine distance;
    lower distance = higher similarity).

    Bug 1C fix: Superseded facts are excluded from query results
    to prevent contradictory information from being injected into RAG context.
    """
    collection = get_collection()

    if collection.count() == 0:
        return []

    # Clamp top_k to available documents
    effective_k = min(top_k, collection.count())

    # Bug 1C: Filter out superseded facts so stale/contradictory data
    # never pollutes the RAG context sent to the LLM
    results = collection.query(
        query_texts=[query],
        n_results=effective_k,
        where={"superseded": {"$ne": "true"}},
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
