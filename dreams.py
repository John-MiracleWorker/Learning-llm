"""
Continual Learning LLM — Dream Cycle & Advanced Training (dreams.py)

Tier 3 creative features:
  1. Dream Cycle         — combine known facts into novel synthetic connections
  2. Meta-Learning Journal — temporal awareness of when things were learned
  3. Style Cloning       — adapt communication patterns to mirror the user
  4. Episodic Memory Compression — consolidate raw facts into coherent profiles

Bug 3A fix: All model-using functions accept model/tokenizer via dependency
injection. The model is loaded ONCE in run_dream_enhanced_sleep() and passed
to all sub-functions, eliminating 3x redundant 4GB SSD→RAM loads.

Bug 3C fix: compress_memories() now processes facts in batches of
MEMORY_COMPRESSION_BATCH_SIZE instead of dumping all facts into one prompt.
"""

import gc
import json
import os
import random
import re
import time
from typing import Optional

import config
import memory


# ── Model Loading Helper ─────────────────────────────────────────────────


def _load_model():
    """Load the base model + adapter (if exists). Shared helper."""
    from mlx_lm import load

    adapter_file = os.path.join(config.ADAPTER_DIR, "adapters.safetensors")
    if os.path.exists(adapter_file):
        return load(
            config.BASE_MODEL,
            adapter_path=config.ADAPTER_DIR,
            tokenizer_config={"trust_remote_code": True},
        )
    else:
        return load(
            config.BASE_MODEL,
            tokenizer_config={"trust_remote_code": True},
        )


# ── Dream Cycle ──────────────────────────────────────────────────────────


def dream_combine_facts(
    n_dreams: int = 10,
    model=None,
    tokenizer=None,
    log_callback: Optional[callable] = None,
) -> list[dict]:
    """
    Generate 'dream' training data by asking the model to combine
    pairs/triples of known facts into novel insights or connections.

    Bug 3A: model/tokenizer are injected from the caller to avoid
    loading 4GB+ weights from SSD multiple times per dream cycle.

    Args:
        n_dreams:     Number of dream sequences to generate.
        model:        Pre-loaded MLX model (injected).
        tokenizer:    Pre-loaded tokenizer (injected).
        log_callback: Optional fn(msg) for logging.

    Returns:
        List of ChatML-formatted training samples.
    """
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    def _log(msg):
        if log_callback:
            log_callback(msg)

    # Gather all learned facts from memory
    collection = memory.get_collection()
    all_facts = collection.get(
        where={"learned": "true"},
        include=["documents"],
    )
    documents = all_facts.get("documents", [])

    if len(documents) < 2:
        _log("[dream] Not enough learned facts for dreaming (need ≥2)")
        return []

    _log(f"[dream] Drawing from {len(documents)} learned facts...")

    # Load model only if not injected (backward compatibility)
    owns_model = False
    if model is None or tokenizer is None:
        model, tokenizer = _load_model()
        owns_model = True

    dream_samples = []

    for i in range(n_dreams):
        # Pick 2-3 random facts to combine
        n_combine = min(random.choice([2, 3]), len(documents))
        selected = random.sample(documents, n_combine)
        combined_text = "\n".join(f"- {f}" for f in selected)

        prompt_messages = [
            {"role": "system", "content": (
                "You are a creative learning assistant. Given a list of known facts, "
                "create ONE question that connects or relates these facts in an interesting way, "
                "then provide a comprehensive answer. Format as JSON: "
                '{"question": "...", "answer": "..."}'
            )},
            {"role": "user", "content": f"Connect these facts:\n{combined_text}"},
        ]
        prompt = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True,
        )
        raw = generate(model, tokenizer, prompt=prompt, max_tokens=512, sampler=make_sampler(temp=0.7))
        raw = re.sub(r"<think>.*?</think>\s*", "", raw, flags=re.DOTALL).strip()

        # Parse the dream
        try:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1:
                dream = json.loads(raw[start:end + 1])
                if "question" in dream and "answer" in dream:
                    sample = {
                        "messages": [
                            {"role": "user", "content": dream["question"]},
                            {"role": "assistant", "content": dream["answer"]},
                        ]
                    }
                    dream_samples.append(sample)
                    _log(f"[dream] 💭 {i+1}/{n_dreams}: {dream['question'][:60]}...")
        except (json.JSONDecodeError, ValueError):
            _log(f"[dream] ⚠️ {i+1}/{n_dreams}: failed to parse dream")

    # Only free model if we loaded it ourselves
    if owns_model:
        del model, tokenizer
        gc.collect()

    _log(f"[dream] Generated {len(dream_samples)} dream sequences")
    return dream_samples


# ── Meta-Learning Journal ────────────────────────────────────────────────

JOURNAL_PATH = os.path.join(config.DATA_DIR, "learning_journal.jsonl")


def log_learning_event(
    fact_text: str,
    fact_id: str,
    event_type: str = "learned",
    metadata: Optional[dict] = None,
):
    """
    Record a timestamped learning event for temporal awareness.

    The model will be trained on journal entries so it can say things like
    "You told me that last Tuesday" or "I learned about X before Y."

    Args:
        fact_text:  The fact that was learned.
        fact_id:    ChromaDB fact ID.
        event_type: "learned", "superseded", "corrected", "dreamed".
        metadata:   Optional extra context.
    """
    entry = {
        "timestamp": time.time(),
        "iso_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "event_type": event_type,
        "fact_id": fact_id,
        "fact_text": fact_text,
    }
    if metadata:
        entry["metadata"] = metadata

    os.makedirs(os.path.dirname(JOURNAL_PATH), exist_ok=True)
    with open(JOURNAL_PATH, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def generate_temporal_training_data(
    log_callback: Optional[callable] = None,
) -> list[dict]:
    """
    Generate training samples that teach temporal awareness.

    Converts journal entries into Q/A pairs like:
    Q: "When did you learn about X?"
    A: "You told me about X on February 14th."

    Returns:
        List of ChatML training samples.
    """
    def _log(msg):
        if log_callback:
            log_callback(msg)

    if not os.path.exists(JOURNAL_PATH):
        _log("[journal] No learning journal found")
        return []

    entries = []
    with open(JOURNAL_PATH, "r") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    if not entries:
        return []

    _log(f"[journal] Processing {len(entries)} journal entries...")

    samples = []
    for entry in entries:
        if entry["event_type"] == "learned":
            # Generate temporal awareness training pair
            fact = entry["fact_text"]
            when = entry["iso_time"]
            samples.append({
                "messages": [
                    {"role": "user", "content": f"When did you learn about: {fact}?"},
                    {"role": "assistant", "content": f"You taught me that on {when}."},
                ]
            })
        elif entry["event_type"] == "superseded":
            fact = entry["fact_text"]
            samples.append({
                "messages": [
                    {"role": "user", "content": f"Did you used to know something different about: {fact}?"},
                    {"role": "assistant", "content": (
                        f"Yes, I had a previous understanding that was updated on {entry['iso_time']}. "
                        f"The newer information superseded the old."
                    )},
                ]
            })

    _log(f"[journal] Generated {len(samples)} temporal training pairs")
    return samples


# ── Style Cloning ────────────────────────────────────────────────────────

STYLE_PATH = os.path.join(config.DATA_DIR, "user_style.jsonl")


def record_user_message(message: str):
    """
    Record user messages for style analysis and cloning.
    These are used to train the model to mirror communication patterns.
    """
    if len(message.split()) < 3:  # Skip very short messages
        return

    entry = {
        "timestamp": time.time(),
        "message": message,
    }
    os.makedirs(os.path.dirname(STYLE_PATH), exist_ok=True)
    with open(STYLE_PATH, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def generate_style_training_data(
    max_samples: int = 20,
    model=None,
    tokenizer=None,
    log_callback: Optional[callable] = None,
) -> list[dict]:
    """
    Analyze user messages and generate training samples that teach
    the model to mirror the user's communication style.

    Bug 3A: model/tokenizer are injected from the caller.

    Returns:
        List of ChatML training samples for style adaptation.
    """
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    def _log(msg):
        if log_callback:
            log_callback(msg)

    if not os.path.exists(STYLE_PATH):
        _log("[style] No user messages recorded yet")
        return []

    messages = []
    with open(STYLE_PATH, "r") as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                messages.append(entry["message"])

    if len(messages) < 5:
        _log("[style] Need at least 5 messages for style analysis")
        return []

    # Take a representative sample
    sample_msgs = random.sample(messages, min(15, len(messages)))
    style_examples = "\n".join(f'- "{m}"' for m in sample_msgs)

    _log(f"[style] Analyzing style from {len(messages)} user messages...")

    # Load model only if not injected
    owns_model = False
    if model is None or tokenizer is None:
        model, tokenizer = _load_model()
        owns_model = True

    # Generate style-adapted responses
    prompt_messages = [
        {"role": "system", "content": (
            "Analyze the user's communication style from these examples, "
            "then generate 5 Q/A pairs where the assistant matches that style. "
            'Output as a JSON array of {"question": ..., "answer": ...} objects. '
            "Match tone, formality, sentence length, and vocabulary."
        )},
        {"role": "user", "content": f"User's communication style:\n{style_examples}"},
    ]
    prompt = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True,
    )
    raw = generate(model, tokenizer, prompt=prompt, max_tokens=1024, sampler=make_sampler(temp=0.5))
    raw = re.sub(r"<think>.*?</think>\s*", "", raw, flags=re.DOTALL).strip()

    if owns_model:
        del model, tokenizer
        gc.collect()

    # Parse style-adapted training pairs
    samples = []
    try:
        start = raw.find("[")
        end = raw.rfind("]")
        if start != -1 and end != -1:
            pairs = json.loads(raw[start:end + 1])
            for pair in pairs[:max_samples]:
                if isinstance(pair, dict) and "question" in pair and "answer" in pair:
                    samples.append({
                        "messages": [
                            {"role": "user", "content": pair["question"]},
                            {"role": "assistant", "content": pair["answer"]},
                        ]
                    })
    except (json.JSONDecodeError, ValueError):
        _log("[style] Failed to parse style training data")

    _log(f"[style] Generated {len(samples)} style-adapted training pairs")
    return samples


# ── Episodic Memory Compression ──────────────────────────────────────────


def compress_memories(
    model=None,
    tokenizer=None,
    log_callback: Optional[callable] = None,
) -> list[dict]:
    """
    Consolidate raw individual facts into coherent knowledge profiles.

    Bug 3A: model/tokenizer are injected from the caller.
    Bug 3C: Facts are processed in batches of MEMORY_COMPRESSION_BATCH_SIZE
    instead of all at once, preventing context window blowout when the user
    has taught hundreds of facts over time.

    Returns:
        List of ChatML training samples representing compressed knowledge.
    """
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    def _log(msg):
        if log_callback:
            log_callback(msg)

    # Gather all learned facts
    collection = memory.get_collection()
    all_data = collection.get(
        where={"learned": "true"},
        include=["documents"],
    )
    documents = all_data.get("documents", [])

    if len(documents) < 5:
        _log("[compress] Not enough facts to compress (need ≥5)")
        return []

    _log(f"[compress] Compressing {len(documents)} facts into profiles...")

    # Load model only if not injected
    owns_model = False
    if model is None or tokenizer is None:
        model, tokenizer = _load_model()
        owns_model = True

    # Bug 3C: Process facts in batches to avoid context window blowout
    batch_size = config.MEMORY_COMPRESSION_BATCH_SIZE
    all_samples = []

    # Shuffle to get diverse batches (approximation of semantic clustering)
    shuffled_docs = documents.copy()
    random.shuffle(shuffled_docs)

    n_batches = (len(shuffled_docs) + batch_size - 1) // batch_size
    _log(f"[compress] Processing in {n_batches} batches of ≤{batch_size} facts...")

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(shuffled_docs))
        batch = shuffled_docs[start:end]

        if len(batch) < 2:
            continue  # Skip tiny batches

        facts_text = "\n".join(f"- {f}" for f in batch)
        prompt_messages = [
            {"role": "system", "content": (
                "You are a knowledge compressor. Given a list of individual facts about a user, "
                "group related facts into coherent profiles and summaries. "
                'Output as a JSON array of {"topic": "...", "summary": "..."} objects. '
                "Each summary should consolidate multiple related facts into one clear paragraph."
            )},
            {"role": "user", "content": f"Compress these facts:\n{facts_text}"},
        ]
        prompt = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True,
        )
        raw = generate(model, tokenizer, prompt=prompt, max_tokens=1024, sampler=make_sampler(temp=0.3))
        raw = re.sub(r"<think>.*?</think>\s*", "", raw, flags=re.DOTALL).strip()

        # Parse compressed profiles into training data
        try:
            start_idx = raw.find("[")
            end_idx = raw.rfind("]")
            if start_idx != -1 and end_idx != -1:
                profiles = json.loads(raw[start_idx:end_idx + 1])
                for profile in profiles:
                    if isinstance(profile, dict) and "topic" in profile and "summary" in profile:
                        all_samples.append({
                            "messages": [
                                {"role": "user", "content": f"What do you know about {profile['topic']}?"},
                                {"role": "assistant", "content": profile["summary"]},
                            ]
                        })
                        _log(f"[compress] 📦 {profile['topic']}")
        except (json.JSONDecodeError, ValueError):
            _log(f"[compress] Failed to parse batch {batch_idx + 1}/{n_batches}")

    if owns_model:
        del model, tokenizer
        gc.collect()

    _log(f"[compress] Generated {len(all_samples)} compressed knowledge profiles")
    return all_samples


# ── Enhanced Sleep Cycle with Dreams ─────────────────────────────────────


def run_dream_enhanced_sleep(
    fact_ids: list[str],
    facts: Optional[list[dict]] = None,
    n_dreams: int = 10,
    include_style: bool = True,
    include_temporal: bool = True,
    include_compression: bool = True,
    log_callback: Optional[callable] = None,
) -> dict:
    """
    Enhanced sleep cycle that includes dream sequences, style adaptation,
    temporal awareness, and memory compression alongside normal training.

    Bug 3A fix: Loads the model ONCE here and passes it to all sub-functions
    via dependency injection, eliminating 3x redundant 4GB SSD→RAM loads
    that caused massive disk I/O bottlenecks and Metal VRAM fragmentation.

    This is called BEFORE trainer.run_sleep_cycle() (Bug 1A fix in app.py)
    so dream data is included in the training dataset.

    Args:
        fact_ids:      IDs of facts to mark as learned.
        facts:         Optional fact dicts for verification.
        n_dreams:      Number of dream sequences to generate.
        include_style: Whether to include style training data.
        include_temporal: Whether to include temporal training data.
        include_compression: Whether to include compressed profiles.
        log_callback:  Optional fn(msg) for logging.

    Returns:
        Dict with enhanced sleep cycle results.
    """
    def _log(msg):
        if log_callback:
            log_callback(msg)

    result = {"dreams": 0, "style": 0, "temporal": 0, "compressed": 0}
    extra_samples = []

    # Bug 3A: Load model ONCE for all dream sub-functions
    _log("[dream] Loading model for dream cycle (single load)...")
    model, tokenizer = _load_model()

    try:
        # Dream cycle
        _log("")
        _log("=" * 60)
        _log("  DREAM PHASE: Knowledge Generalization")
        _log("=" * 60)
        _log("")
        dream_data = dream_combine_facts(
            n_dreams=n_dreams,
            model=model, tokenizer=tokenizer,
            log_callback=log_callback,
        )
        extra_samples.extend(dream_data)
        result["dreams"] = len(dream_data)

        # Style cloning
        if include_style:
            _log("")
            _log("── Style Adaptation ──")
            style_data = generate_style_training_data(
                model=model, tokenizer=tokenizer,
                log_callback=log_callback,
            )
            extra_samples.extend(style_data)
            result["style"] = len(style_data)

        # Temporal awareness (no model needed — pure data transform)
        if include_temporal:
            _log("")
            _log("── Temporal Awareness ──")
            temporal_data = generate_temporal_training_data(log_callback=log_callback)
            extra_samples.extend(temporal_data)
            result["temporal"] = len(temporal_data)

        # Memory compression
        if include_compression:
            _log("")
            _log("── Memory Compression ──")
            compressed_data = compress_memories(
                model=model, tokenizer=tokenizer,
                log_callback=log_callback,
            )
            extra_samples.extend(compressed_data)
            result["compressed"] = len(compressed_data)

    finally:
        # Bug 3A: Free model after all sub-functions complete
        del model, tokenizer
        gc.collect()
        try:
            import mlx.core as mx
            mx.metal.clear_cache()
        except (ImportError, AttributeError):
            pass

    # Append extra training samples to the training data
    if extra_samples and os.path.exists(config.TRAIN_JSONL_PATH):
        _log(f"\n[dream] Appending {len(extra_samples)} enhanced samples to training data...")
        with open(config.TRAIN_JSONL_PATH, "a") as f:
            for sample in extra_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # Log learning events for temporal awareness
    if facts:
        for fact in facts:
            log_learning_event(
                fact_text=fact.get("text", ""),
                fact_id=fact.get("id", ""),
                event_type="learned",
            )

    _log(f"\n[dream] Enhanced data: {result['dreams']} dreams, "
         f"{result['style']} style, {result['temporal']} temporal, "
         f"{result['compressed']} compressed")

    result["total_extra_samples"] = len(extra_samples)
    return result
