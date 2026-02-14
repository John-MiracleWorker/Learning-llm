"""
Continual Learning LLM — Sanitizer & Mixer (curator.py)

Bridges the Wake → Sleep transition:
  1. Extract unlearned facts from ChromaDB
  2. Use the Qwen model (via mlx_lm) to synthesize 3–5 diverse Q/A pairs
     per fact in ChatML JSONL format
  3. Verify synthesized QA pairs for hallucinations (Bug 2C)
  4. Append ONLY new samples to Replay Buffer (Bug 1B fix)
  5. Mathematically mix new data with the Replay Buffer at 20/80 ratio
  6. Enforce MINIMUM_DATASET_SIZE to prevent overfitting (Bug 2A)
  7. Write train.jsonl + valid.jsonl for the Sleep phase
"""

import gc
import json
import math
import os
import random
import re
from typing import Optional

import config
import memory


# ── Synthetic QA Generation Prompt ────────────────────────────────────────

_SYNTHESIS_PROMPT = """\
You are a dataset generator. Given a FACT, produce exactly {n} diverse \
question/answer pairs that test knowledge of this fact from different angles.

Rules:
- Each question must be meaningfully different (rephrase, ask from a \
different perspective, ask about implications, etc.)
- Answers must be self-contained, accurate, and 1-3 sentences long.
- Output ONLY a JSON array of objects with "question" and "answer" keys.
- No markdown fences, no commentary — just the raw JSON array.

FACT: {fact}

JSON array of {n} Q/A pairs:"""


def _parse_qa_pairs(raw_output: str) -> list[dict]:
    """
    Parse the LLM output into a list of {"question": ..., "answer": ...} dicts.
    Handles common LLM quirks: markdown fences, trailing commas, preamble text.
    """
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?\s*", "", raw_output)
    cleaned = cleaned.strip()

    # Try to find the JSON array in the output
    # Look for the outermost [ ... ]
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []

    json_str = cleaned[start:end + 1]

    # Fix trailing commas before ] (common LLM mistake)
    json_str = re.sub(r",\s*]", "]", json_str)
    json_str = re.sub(r",\s*}", "}", json_str)

    try:
        pairs = json.loads(json_str)
    except json.JSONDecodeError:
        return []

    # Validate structure
    validated = []
    for pair in pairs:
        if isinstance(pair, dict) and "question" in pair and "answer" in pair:
            validated.append({
                "question": str(pair["question"]).strip(),
                "answer": str(pair["answer"]).strip(),
            })
    return validated


def _qa_to_chatml(question: str, answer: str) -> dict:
    """Convert a Q/A pair to Qwen ChatML JSONL format."""
    return {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }


# ── Hallucination Verification (Bug 2C) ──────────────────────────────────


def _verify_qa_pair(
    fact: str,
    question: str,
    answer: str,
    model,
    tokenizer,
) -> bool:
    """
    Verify that a generated Q/A pair does not hallucinate beyond the source fact.

    The 4B model generating its own training data can introduce errors
    that get permanently baked into the replay buffer, causing Model Collapse.

    Args:
        fact:      The source fact the QA pair was generated from.
        question:  The generated question.
        answer:    The generated answer.
        model:     Pre-loaded MLX model.
        tokenizer: Pre-loaded tokenizer.

    Returns:
        True if the QA pair is faithful to the fact (passes verification).
    """
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    messages = [
        {"role": "system", "content": (
            "You are a strict fact-checker. Determine whether the ANSWER "
            "contains information NOT present in or directly inferable from the FACT. "
            "Answer ONLY 'YES' (contains hallucination) or 'NO' (faithful to fact)."
        )},
        {"role": "user", "content": (
            f'FACT: "{fact}"\n'
            f'QUESTION: "{question}"\n'
            f'ANSWER: "{answer}"\n\n'
            "Does the ANSWER contain hallucinated information not present in the FACT? (YES/NO)"
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

    # "YES" means hallucination detected → pair FAILS verification
    return "YES" not in raw.upper()


# ── Core Pipeline ─────────────────────────────────────────────────────────


def generate_qa_pairs(
    fact: str,
    n_pairs: int = 4,
    model=None,
    tokenizer=None,
    verify: bool = True,
) -> list[dict]:
    """
    Use the loaded Qwen model to synthesize diverse Q/A pairs for a fact.

    Args:
        fact:      The raw fact text.
        n_pairs:   Number of Q/A pairs to generate (3–5).
        model:     Pre-loaded MLX model (if None, will load).
        tokenizer: Pre-loaded tokenizer (if None, will load).
        verify:    If True, run hallucination check on each pair (Bug 2C).

    Returns:
        List of ChatML-formatted dicts ready for JSONL.
    """
    from mlx_lm import generate, load

    if model is None or tokenizer is None:
        model, tokenizer = load(
            config.BASE_MODEL,
            tokenizer_config={"trust_remote_code": True},
        )

    prompt = _SYNTHESIS_PROMPT.format(fact=fact, n=n_pairs)

    # Build chat messages for the model
    messages = [
        {"role": "system", "content": "You are a precise dataset generator."},
        {"role": "user", "content": prompt},
    ]
    chat_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    from mlx_lm.sample_utils import make_sampler

    raw_output = generate(
        model,
        tokenizer,
        prompt=chat_prompt,
        max_tokens=config.MAX_TOKENS,
        sampler=make_sampler(temp=0.7),
    )

    # Strip <think>...</think> blocks from Qwen3 thinking model output
    raw_output = re.sub(r"<think>.*?</think>\s*", "", raw_output, flags=re.DOTALL)
    pairs = _parse_qa_pairs(raw_output)

    # Convert to ChatML format with optional hallucination verification
    chatml_pairs = []
    for pair in pairs:
        if verify:
            # Bug 2C: Verify the QA pair doesn't hallucinate beyond the fact
            if not _verify_qa_pair(fact, pair["question"], pair["answer"], model, tokenizer):
                continue  # Discard hallucinated pairs
        chatml_pairs.append(_qa_to_chatml(pair["question"], pair["answer"]))

    return chatml_pairs


def synthesize_all_facts(
    facts: list[dict],
    n_pairs: int = 4,
    progress_callback: Optional[callable] = None,
) -> list[dict]:
    """
    Generate synthetic Q/A pairs for all provided facts.

    Args:
        facts:             List of fact dicts from memory.get_unlearned_facts().
        n_pairs:           Q/A pairs per fact.
        progress_callback: Optional fn(current_idx, total, fact_text) for UI.

    Returns:
        List of all ChatML-formatted training samples.
    """
    from mlx_lm import load

    # Load model once for all facts
    model, tokenizer = load(
        config.BASE_MODEL,
        tokenizer_config={"trust_remote_code": True},
    )

    all_samples = []
    total = len(facts)

    for idx, fact in enumerate(facts):
        if progress_callback:
            progress_callback(idx, total, fact["text"])

        pairs = generate_qa_pairs(
            fact["text"],
            n_pairs=n_pairs,
            model=model,
            tokenizer=tokenizer,
            verify=True,  # Bug 2C: verify each pair
        )
        all_samples.extend(pairs)

    # Free the model from unified memory
    del model, tokenizer
    gc.collect()

    if progress_callback:
        progress_callback(total, total, "Done")

    return all_samples


# ── Replay Buffer Mixing (CRITICAL MATH) ─────────────────────────────────


def mix_with_replay(new_samples: list[dict]) -> list[dict]:
    """
    Mix new daily samples with the Replay Buffer at the configured ratio.

    Math:
        If we have N new samples, the total dataset size T must satisfy:
            N / T = NEW_DATA_RATIO  →  T = N / NEW_DATA_RATIO
        Old samples needed: T - N = N * (OLD_DATA_RATIO / NEW_DATA_RATIO)

        If the replay buffer has fewer samples than needed, we oversample
        (repeat) from the buffer to maintain the exact ratio.

    Bug 2A fix: If the mixed total is below MINIMUM_DATASET_SIZE, pad with
    additional unique replay buffer samples to prevent overfitting.
    """
    if not new_samples:
        return []

    # Load the replay buffer
    replay = []
    if os.path.exists(config.REPLAY_BUFFER_PATH):
        with open(config.REPLAY_BUFFER_PATH, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    replay.append(json.loads(line))

    if not replay:
        # No replay buffer — use only new data (not ideal, but functional)
        random.shuffle(new_samples)
        return new_samples

    n_new = len(new_samples)

    # Calculate required old samples to maintain 20/80 ratio
    # T = N / 0.20, old = T - N = N * (0.80 / 0.20) = N * 4
    n_old_needed = math.ceil(n_new * (config.OLD_DATA_RATIO / config.NEW_DATA_RATIO))

    # Bug 2A: Enforce minimum dataset size to prevent extreme overfitting
    # If the 20/80 math yields fewer than MINIMUM_DATASET_SIZE total samples,
    # pad exclusively with unique replay buffer data to anchor LoRA weights.
    total_before_padding = n_new + n_old_needed
    if total_before_padding < config.MINIMUM_DATASET_SIZE:
        n_old_needed = config.MINIMUM_DATASET_SIZE - n_new

    # Oversample from replay buffer if needed
    old_samples = []
    while len(old_samples) < n_old_needed:
        shuffled_replay = replay.copy()
        random.shuffle(shuffled_replay)
        old_samples.extend(shuffled_replay)
    old_samples = old_samples[:n_old_needed]

    # Combine and shuffle
    mixed = new_samples + old_samples
    random.shuffle(mixed)

    return mixed


# ── Dataset Writing ───────────────────────────────────────────────────────


def write_training_data(
    samples: list[dict],
    valid_split: float = 0.1,
) -> tuple[int, int]:
    """
    Write shuffled samples to train.jsonl and valid.jsonl.

    Args:
        samples:     The mixed (new + replay) dataset.
        valid_split: Fraction reserved for validation.

    Returns:
        (train_count, valid_count)
    """
    random.shuffle(samples)

    split_idx = max(1, int(len(samples) * (1 - valid_split)))
    train_data = samples[:split_idx]
    valid_data = samples[split_idx:]

    # Ensure at least 1 validation sample
    if not valid_data and len(train_data) > 1:
        valid_data = [train_data.pop()]

    os.makedirs(config.DATA_DIR, exist_ok=True)

    with open(config.TRAIN_JSONL_PATH, "w") as f:
        for sample in train_data:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    with open(config.VALID_JSONL_PATH, "w") as f:
        for sample in valid_data:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    return len(train_data), len(valid_data)


# ── Full Pipeline ─────────────────────────────────────────────────────────


def run_sanitize_pipeline(
    n_pairs: int = 4,
    progress_callback: Optional[callable] = None,
) -> dict:
    """
    Execute the full Sanitize phase:
      1. Pull unlearned facts from ChromaDB
      2. Synthesize Q/A pairs (with hallucination verification)
      3. Append ONLY new samples to replay buffer (Bug 1B fix)
      4. Mix with replay buffer (20/80) with minimum dataset enforcement
      5. Write train.jsonl + valid.jsonl

    Args:
        n_pairs:           Q/A pairs to generate per fact.
        progress_callback: Optional fn(idx, total, text) for progress.

    Returns:
        Dict with pipeline stats:
          facts_processed, qa_generated, replay_mixed,
          train_count, valid_count, fact_ids
    """
    # 1. Extract unlearned facts
    facts = memory.get_unlearned_facts()
    if not facts:
        return {
            "facts_processed": 0,
            "qa_generated": 0,
            "replay_mixed": 0,
            "train_count": 0,
            "valid_count": 0,
            "fact_ids": [],
        }

    # 2. Synthesize (with hallucination verification — Bug 2C)
    new_samples = synthesize_all_facts(
        facts, n_pairs=n_pairs, progress_callback=progress_callback,
    )

    # 3. Bug 1B fix: Append ONLY the newly synthesized samples to the
    #    replay buffer BEFORE mixing. This ensures each new QA pair enters
    #    the buffer exactly once. Previously, the entire mixed dataset
    #    (including the 80% old data) was re-appended in trainer.py,
    #    causing exponential duplication.
    if new_samples:
        os.makedirs(config.DATA_DIR, exist_ok=True)
        with open(config.REPLAY_BUFFER_PATH, "a") as f:
            for sample in new_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # 4. Mix with replay buffer (with minimum dataset enforcement — Bug 2A)
    mixed = mix_with_replay(new_samples)

    # 5. Write to disk
    train_count, valid_count = write_training_data(mixed)

    # Collect fact IDs for marking as learned after training
    fact_ids = [f["id"] for f in facts]

    return {
        "facts_processed": len(facts),
        "qa_generated": len(new_samples),
        "replay_mixed": len(mixed),
        "train_count": train_count,
        "valid_count": valid_count,
        "fact_ids": fact_ids,
        "facts": facts,
    }
