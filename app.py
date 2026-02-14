"""
Continual Learning LLM — Streamlit Frontend (app.py)

Three-tab interface:
  1. Chat UI      — RAG-augmented conversation with 🧠 indicator
  2. Memory Admin — View/delete unlearned facts in ChromaDB
  3. Sleep Cycle  — Trigger Sanitize → Train pipeline with live logs

Bug 1A fix: Dream cycle runs BEFORE training so dream data is included.
Bug 3B fix: Training is decoupled from the UI via worker.py flag/status
pattern, preventing WebSocket timeout on long-running training.
"""

from __future__ import annotations

import gc
import json
import os
import time
from datetime import datetime

import streamlit as st

import re

import config
import memory


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from Qwen3 thinking model output."""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()

# ── Page Config ───────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Continual Learning LLM",
    page_icon="🧠",
    layout="wide",
)

# ── Session State Initialization ──────────────────────────────────────────

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "model" not in st.session_state:
    st.session_state.model = None
    st.session_state.tokenizer = None

if "sleep_log" not in st.session_state:
    st.session_state.sleep_log = []

if "sleep_running" not in st.session_state:
    st.session_state.sleep_running = False

if "sleep_result" not in st.session_state:
    st.session_state.sleep_result = None


# ── Model Loading ─────────────────────────────────────────────────────────


def load_model():
    """Load the base model + LoRA adapter (if exists) into unified memory."""
    if st.session_state.model is not None:
        return st.session_state.model, st.session_state.tokenizer

    from mlx_lm import load

    adapter_path = config.ADAPTER_DIR
    adapter_file = os.path.join(adapter_path, "adapters.safetensors")

    if os.path.exists(adapter_file):
        # Load base model with LoRA adapter
        model, tokenizer = load(
            config.BASE_MODEL,
            adapter_path=adapter_path,
            tokenizer_config={"trust_remote_code": True},
        )
    else:
        # Load base model only (no adapter yet)
        model, tokenizer = load(
            config.BASE_MODEL,
            tokenizer_config={"trust_remote_code": True},
        )

    st.session_state.model = model
    st.session_state.tokenizer = tokenizer
    return model, tokenizer


def unload_model():
    """Free model from unified memory (before Sleep cycle)."""
    st.session_state.model = None
    st.session_state.tokenizer = None
    gc.collect()
    try:
        import mlx.core as mx
        mx.metal.clear_cache()
    except (ImportError, AttributeError):
        pass
    gc.collect()


# ── Inference ─────────────────────────────────────────────────────────────


def generate_response(user_message: str) -> tuple[str, bool]:
    """
    Generate a response with RAG-augmented context.

    Returns:
        (response_text, used_rag) — the model's reply and whether
        short-term memory facts were injected.
    """
    from mlx_lm import generate

    model, tokenizer = load_model()

    # Build RAG context from ChromaDB
    rag_context, has_rag = memory.build_rag_context(user_message)

    # System prompt with optional RAG injection
    system_content = "You are a helpful assistant."
    if has_rag:
        system_content += (
            "\n\nYou have access to the following recently learned facts. "
            "Use them to inform your answer when relevant. "
            "If the facts contradict your training data, prefer the facts.\n\n"
            + rag_context
        )

    messages = [{"role": "system", "content": system_content}]

    # Add conversation history (last 10 turns for context window management)
    for msg in st.session_state.chat_history[-10:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # Add current user message
    messages.append({"role": "user", "content": user_message})

    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    from mlx_lm.sample_utils import make_sampler
    sampler = make_sampler(temp=config.TEMPERATURE, top_p=config.TOP_P)

    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=config.MAX_TOKENS,
        sampler=sampler,
    )

    return strip_thinking(response), has_rag


# ── Fact Detection Heuristic ──────────────────────────────────────────────

_FACT_PREFIXES = [
    "remember that",
    "remember:",
    "learn that",
    "learn:",
    "fact:",
    "note that",
    "note:",
    "correction:",
    "actually,",
    "update:",
    "fyi:",
]


def detect_fact(message: str) -> tuple[bool, str]:
    """
    Check if the user is teaching a new fact or correction.

    Returns:
        (is_fact, extracted_fact_text)
    """
    lower = message.lower().strip()
    for prefix in _FACT_PREFIXES:
        if lower.startswith(prefix):
            # Extract the fact after the prefix
            fact = message[len(prefix):].strip()
            if fact:
                return True, fact
    return False, ""


def extract_facts_from_message(message: str) -> list[str]:
    """
    Use the LLM to detect learnable personal/factual info from natural chat.

    For example, if the user says "I work as a nurse at Memorial Hospital",
    this extracts: ["The user works as a nurse", "The user works at Memorial Hospital"]

    Returns:
        List of extracted fact strings, or empty list if none found.
    """
    # Skip short messages or questions
    if len(message.split()) < 4 or message.strip().endswith("?"):
        return []

    from mlx_lm import generate

    model, tokenizer = load_model()

    prompt_messages = [
        {"role": "system", "content": (
            "You are a fact extractor. Given a user message, extract any personal "
            "facts, preferences, or corrections that would be useful to remember. "
            "Output ONLY a JSON array of strings. If no facts are found, output []. "
            "Examples of facts: name, job, preferences, family, location, habits. "
            "Do NOT extract questions or generic statements."
        )},
        {"role": "user", "content": f"Extract facts from: \"{message}\""},
    ]
    prompt = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True,
    )
    from mlx_lm.sample_utils import make_sampler
    raw = generate(model, tokenizer, prompt=prompt, max_tokens=256, sampler=make_sampler(temp=0.1))
    raw = strip_thinking(raw)

    # Parse JSON array from response
    try:
        # Find the JSON array in the response
        start = raw.find("[")
        end = raw.rfind("]")
        if start != -1 and end != -1 and end > start:
            facts = json.loads(raw[start:end + 1])
            if isinstance(facts, list):
                return [str(f).strip() for f in facts if isinstance(f, str) and f.strip()]
    except (json.JSONDecodeError, ValueError):
        pass
    return []


# ══════════════════════════════════════════════════════════════════════════
# TAB 1: CHAT UI
# ══════════════════════════════════════════════════════════════════════════


def render_chat_tab():
    st.header("Chat")

    # Sidebar stats
    with st.sidebar:
        st.markdown("### Memory Status")
        total = memory.count()
        unlearned = memory.count_unlearned()
        st.metric("Facts in Memory", total)
        st.metric("Pending (Unlearned)", unlearned)

        adapter_file = os.path.join(config.ADAPTER_DIR, "adapters.safetensors")
        if os.path.exists(adapter_file):
            size_mb = round(os.path.getsize(adapter_file) / (1024 * 1024), 2)
            mod_time = datetime.fromtimestamp(
                os.path.getmtime(adapter_file)
            ).strftime("%Y-%m-%d %H:%M")
            st.success(f"Adapter loaded ({size_mb} MB)")
            st.caption(f"Last trained: {mod_time}")
        else:
            st.info("No adapter yet — base model only")

        st.divider()
        st.caption(f"Model: `{config.BASE_MODEL}`")
        st.caption(f"LoRA: r={config.LORA_RANK}, α={config.LORA_ALPHA}")

    # Teach fact instruction
    st.caption(
        'Tip: Start a message with "Remember that ..." or "Fact: ..." '
        "to teach me something new."
    )

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            prefix = ""
            if msg.get("used_rag"):
                prefix = "🧠 "
            st.markdown(prefix + msg["content"])

    # Chat input
    if user_input := st.chat_input("Ask me anything..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)

        # Check if user is teaching a fact
        is_fact, fact_text = detect_fact(user_input)

        if is_fact:
            # Store the fact in short-term memory (with conflict detection)
            fact_id, superseded = memory.store_fact(fact_text)
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input,
            })
            ack = (
                f"Got it! I've stored this in my short-term memory:\n\n"
                f"> {fact_text}\n\n"
            )
            if superseded:
                old_texts = ", ".join(f'"{s["text"]}"' for s in superseded)
                ack += f"⚠️ This supersedes a previous fact: {old_texts}\n\n"
            ack += (
                "This fact will be available immediately via RAG, and "
                "permanently learned after the next Sleep cycle."
            )
            with st.chat_message("assistant"):
                st.markdown("🧠 " + ack)
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": ack,
                "used_rag": True,
            })
        else:
            # Normal chat — generate response with RAG
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input,
            })
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response, used_rag = generate_response(user_input)
                prefix = "🧠 " if used_rag else ""
                st.markdown(prefix + response)
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response,
                "used_rag": used_rag,
            })

            # Auto-fact extraction: detect learnable info from natural chat
            auto_facts = extract_facts_from_message(user_input)
            if auto_facts:
                for af in auto_facts:
                    memory.store_fact(af, source="auto")
                with st.chat_message("assistant"):
                    st.caption(
                        f"💡 Auto-detected {len(auto_facts)} fact(s) from your message: "
                        + "; ".join(f'"{f}"' for f in auto_facts)
                    )

            # Record user message for style cloning
            try:
                import dreams
                dreams.record_user_message(user_input)
            except ImportError:
                pass

# ══════════════════════════════════════════════════════════════════════════
# TAB 2: SHORT-TERM MEMORY ADMIN
# ══════════════════════════════════════════════════════════════════════════


def render_memory_tab():
    st.header("Short-Term Memory (Admin)")
    st.caption(
        "Review facts stored in ChromaDB. Delete hallucinated or incorrect "
        "facts before they are permanently learned in the Sleep cycle."
    )

    facts = memory.get_all_facts()

    if not facts:
        st.info("No facts in short-term memory yet. Teach me something in the Chat tab!")
        return

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    total = len(facts)
    unlearned = sum(1 for f in facts if f["metadata"].get("learned") == "false")
    learned = total - unlearned
    col1.metric("Total Facts", total)
    col2.metric("Pending (Unlearned)", unlearned)
    col3.metric("Learned (Baked)", learned)

    st.divider()

    # Fact table with delete buttons
    for fact in facts:
        is_learned = fact["metadata"].get("learned") == "true"
        status = "Learned" if is_learned else "Pending"
        badge_color = "green" if is_learned else "orange"

        with st.container():
            col_text, col_meta, col_action = st.columns([5, 2, 1])

            with col_text:
                st.markdown(f"**{fact['text']}**")

            with col_meta:
                st.caption(f"Status: :{badge_color}[{status}]")
                source = fact["metadata"].get("source", "unknown")
                st.caption(f"Source: {source}")
                ts = fact["metadata"].get("timestamp")
                if ts:
                    dt = datetime.fromtimestamp(float(ts)).strftime("%m/%d %H:%M")
                    st.caption(f"Added: {dt}")

            with col_action:
                if st.button("Delete", key=f"del_{fact['id']}", type="secondary"):
                    memory.delete_fact(fact["id"])
                    st.rerun()

        st.divider()

    # Bulk actions
    st.subheader("Bulk Actions")
    col_a, col_b = st.columns(2)

    with col_a:
        if st.button("Wipe All Learned Facts", type="secondary"):
            wiped = memory.wipe_learned()
            st.success(f"Removed {wiped} learned facts from ChromaDB.")
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# TAB 3: SLEEP CYCLE ADMIN (Bug 3B: Non-blocking worker pattern)
# ══════════════════════════════════════════════════════════════════════════

# Worker paths
_FLAG_PATH = os.path.join(config.DATA_DIR, "start_training.flag")
_STATUS_PATH = os.path.join(config.DATA_DIR, "training_status.json")


def _read_worker_status() -> dict | None:
    """Read the worker's status JSON file."""
    if not os.path.exists(_STATUS_PATH):
        return None
    try:
        with open(_STATUS_PATH, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def _is_worker_running() -> bool:
    """Check if there's an active training run in progress."""
    status = _read_worker_status()
    return status is not None and status.get("state") == "running"


def render_sleep_tab():
    st.header("Sleep Cycle (Admin)")
    st.caption(
        "Trigger the Sanitize → Dream → Train pipeline. This will:\n"
        "1. Generate synthetic Q/A pairs from unlearned facts\n"
        "2. Mix with the Replay Buffer (20% new / 80% old)\n"
        "3. Optionally generate dream/style/temporal training data\n"
        "4. Run MLX LoRA fine-tuning on the M4 GPU\n"
        "5. Save adapter weights (never fuse)"
    )

    # Pre-flight checks
    unlearned = memory.count_unlearned()
    adapter_info = _get_adapter_info()

    col1, col2, col3 = st.columns(3)
    col1.metric("Unlearned Facts", unlearned)
    col2.metric("Replay Buffer",
                f"{_count_replay_buffer()} samples")
    if adapter_info["exists"]:
        col3.metric("Current Adapter",
                     f"{adapter_info['size_mb']} MB")
    else:
        col3.metric("Current Adapter", "None")

    st.divider()

    # Training config preview
    with st.expander("Training Configuration Preview"):
        st.json({
            "base_model": config.BASE_MODEL,
            "lora_rank": config.LORA_RANK,
            "lora_alpha": config.LORA_ALPHA,
            "lora_scale": config.LORA_ALPHA / config.LORA_RANK,
            "lora_dropout": config.LORA_DROPOUT,
            "target_modules": config.LORA_TARGET_MODULES,
            "learning_rate": config.LEARNING_RATE,
            "batch_size": config.BATCH_SIZE,
            "grad_accumulation": config.GRAD_ACCUMULATION,
            "min_iterations": config.MIN_ITERATIONS,
            "min_dataset_size": config.MINIMUM_DATASET_SIZE,
            "replay_ratio": f"{int(config.NEW_DATA_RATIO*100)}% new / "
                           f"{int(config.OLD_DATA_RATIO*100)}% old",
        })

    # Dream toggle
    dream_enabled = st.toggle("🌙 Dream Enhanced Sleep", value=False,
                              help="Include Dream Cycle, Style Adaptation, Temporal Awareness, and Memory Compression")
    st.session_state.dream_enabled = dream_enabled

    # Check current worker status
    worker_status = _read_worker_status()
    worker_active = worker_status is not None and worker_status.get("state") == "running"

    if unlearned == 0 and not worker_active:
        st.warning("No unlearned facts to process. Teach me something in the Chat tab first!")
        trigger_disabled = True
    else:
        trigger_disabled = worker_active

    # Bug 3B: Non-blocking trigger — writes flag file for worker.py
    if st.button(
        "Trigger Sleep Cycle" if not worker_active else "Training in progress...",
        type="primary",
        disabled=trigger_disabled,
        use_container_width=True,
    ):
        _trigger_worker_sleep(dream_enabled)
        st.rerun()

    # Bug 3B: Display live status from worker.py
    if worker_status:
        _render_worker_status(worker_status)

    # Auto-refresh while training is running
    if worker_active:
        time.sleep(3)
        st.rerun()


def _trigger_worker_sleep(dream_enabled: bool):
    """
    Bug 3B: Write a flag file for the background worker instead of
    blocking the Streamlit thread.

    If the worker is not running, fall back to running inline (legacy mode).
    """
    # Unload the inference model to free unified memory for training
    unload_model()

    flag = {
        "dream_enabled": dream_enabled,
        "triggered_at": datetime.now().isoformat(),
    }
    os.makedirs(config.DATA_DIR, exist_ok=True)
    with open(_FLAG_PATH, "w") as f:
        json.dump(flag, f)

    # Write initial status
    with open(_STATUS_PATH, "w") as f:
        json.dump({
            "state": "running",
            "phase": "starting",
            "started_at": datetime.now().isoformat(),
            "log": ["[app] Sleep cycle triggered, waiting for worker..."],
        }, f)


def _render_worker_status(status: dict):
    """Display the worker's current status in the UI."""
    state = status.get("state", "unknown")

    if state == "running":
        phase = status.get("phase", "unknown")
        st.info(f"🔄 Training in progress — Phase: **{phase}**")
        # Show recent log lines
        log_lines = status.get("log", [])
        if log_lines:
            st.subheader("Training Log (Live)")
            st.code("\n".join(log_lines[-30:]), language="text")

    elif state == "completed":
        success = status.get("success", False)
        if success:
            st.success("✅ Sleep cycle completed successfully!")
            # Show results
            sanitize = status.get("sanitize", {})
            training = status.get("training", {})
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Facts Processed", sanitize.get("facts_processed", 0))
            col_b.metric("QA Pairs Generated", sanitize.get("qa_generated", 0))
            col_c.metric("Training Samples", sanitize.get("train_count", 0))

            # Dream results
            dream_result = status.get("dreams")
            if dream_result:
                st.caption(
                    f"🌙 Dream data: {dream_result.get('dreams', 0)} dreams, "
                    f"{dream_result.get('style', 0)} style, "
                    f"{dream_result.get('temporal', 0)} temporal, "
                    f"{dream_result.get('compressed', 0)} compressed"
                )

            # Verification results
            verification = training.get("verification")
            if verification:
                st.subheader("Fact Recall Verification")
                rate = verification["recall_rate"]
                if rate >= 0.7:
                    st.success(f"Recall rate: {rate:.0%} ({verification['passed']}/{verification['passed'] + verification['failed']} facts verified)")
                elif rate >= 0.4:
                    st.warning(f"Recall rate: {rate:.0%} — some facts may need re-training")
                else:
                    st.error(f"Recall rate: {rate:.0%} — consider another sleep cycle")

                for detail in verification.get("details", []):
                    icon = "✅" if detail["status"] == "PASS" else "❌"
                    st.caption(f"{icon} {detail['fact']} — {detail['recall']:.0%}")

        else:
            reason = status.get("reason", "unknown")
            if reason == "no_facts":
                st.warning("No unlearned facts to process.")
            else:
                st.error("Sleep cycle failed. Check the log below for details.")

        # Show log
        log_lines = status.get("log", [])
        if log_lines:
            with st.expander("Training Log"):
                st.code("\n".join(log_lines), language="text")

        # Clear button
        if st.button("Dismiss Results", type="secondary"):
            if os.path.exists(_STATUS_PATH):
                os.remove(_STATUS_PATH)
            st.rerun()

    elif state == "failed":
        st.error(f"❌ Sleep cycle failed: {status.get('error', 'Unknown error')}")
        log_lines = status.get("log", [])
        if log_lines:
            with st.expander("Error Log"):
                st.code("\n".join(log_lines), language="text")

        if st.button("Dismiss Error", type="secondary"):
            if os.path.exists(_STATUS_PATH):
                os.remove(_STATUS_PATH)
            st.rerun()


# ── Helpers ───────────────────────────────────────────────────────────────


def _get_adapter_info() -> dict:
    """Check adapter status."""
    adapter_file = os.path.join(config.ADAPTER_DIR, "adapters.safetensors")
    exists = os.path.exists(adapter_file)
    info = {"exists": exists}
    if exists:
        info["size_mb"] = round(os.path.getsize(adapter_file) / (1024 * 1024), 2)
        info["modified"] = datetime.fromtimestamp(
            os.path.getmtime(adapter_file)
        ).strftime("%Y-%m-%d %H:%M")
    return info


def _count_replay_buffer() -> int:
    """Count samples in the replay buffer."""
    if not os.path.exists(config.REPLAY_BUFFER_PATH):
        return 0
    with open(config.REPLAY_BUFFER_PATH, "r") as f:
        return sum(1 for line in f if line.strip())


# ── Main Layout ───────────────────────────────────────────────────────────


def main():
    st.title("Continual Learning LLM")
    st.caption("Sleep/Wake Cycle on Apple M4 — Powered by MLX + Qwen 3")

    tab_chat, tab_memory, tab_sleep = st.tabs([
        "💬 Chat",
        "📋 Short-Term Memory",
        "🌙 Sleep Cycle",
    ])

    with tab_chat:
        render_chat_tab()

    with tab_memory:
        render_memory_tab()

    with tab_sleep:
        render_sleep_tab()


if __name__ == "__main__":
    main()
