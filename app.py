"""
Continual Learning LLM — Streamlit Frontend (app.py)

Three-tab interface:
  1. Chat UI      — RAG-augmented conversation with 🧠 indicator
  2. Memory Admin — View/delete unlearned facts in ChromaDB
  3. Sleep Cycle  — Trigger Sanitize → Train pipeline with live logs
"""

import gc
import os
import time
from datetime import datetime

import streamlit as st

import config
import memory

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

    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=config.MAX_TOKENS,
        temp=config.TEMPERATURE,
        top_p=config.TOP_P,
        repetition_penalty=config.REPETITION_PENALTY,
    )

    return response.strip(), has_rag


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
            # Store the fact in short-term memory
            fact_id = memory.store_fact(fact_text)
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input,
            })
            ack = (
                f"Got it! I've stored this in my short-term memory:\n\n"
                f"> {fact_text}\n\n"
                f"This fact will be available immediately via RAG, and "
                f"permanently learned after the next Sleep cycle."
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
# TAB 3: SLEEP CYCLE ADMIN
# ══════════════════════════════════════════════════════════════════════════


def render_sleep_tab():
    st.header("Sleep Cycle (Admin)")
    st.caption(
        "Trigger the Sanitize → Train pipeline. This will:\n"
        "1. Generate synthetic Q/A pairs from unlearned facts\n"
        "2. Mix with the Replay Buffer (20% new / 80% old)\n"
        "3. Run MLX LoRA fine-tuning on the M4 GPU\n"
        "4. Save adapter weights (never fuse)"
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
            "replay_ratio": f"{int(config.NEW_DATA_RATIO*100)}% new / "
                           f"{int(config.OLD_DATA_RATIO*100)}% old",
        })

    # Trigger button
    if unlearned == 0:
        st.warning("No unlearned facts to process. Teach me something in the Chat tab first!")
        trigger_disabled = True
    else:
        trigger_disabled = st.session_state.sleep_running

    if st.button(
        "Trigger Sleep Cycle",
        type="primary",
        disabled=trigger_disabled,
        use_container_width=True,
    ):
        _run_full_sleep_cycle()

    # Display log output
    if st.session_state.sleep_log:
        st.subheader("Training Log")
        log_text = "\n".join(st.session_state.sleep_log)
        st.code(log_text, language="text")

    # Display result
    if st.session_state.sleep_result:
        result = st.session_state.sleep_result
        if result.get("success"):
            st.success("Sleep cycle completed successfully!")
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Facts Processed",
                         result.get("sanitize", {}).get("facts_processed", 0))
            col_b.metric("QA Pairs Generated",
                         result.get("sanitize", {}).get("qa_generated", 0))
            col_c.metric("Training Samples",
                         result.get("sanitize", {}).get("train_count", 0))
        else:
            st.error("Sleep cycle failed. Check the log above for details.")


def _run_full_sleep_cycle():
    """Execute Sanitize → Sleep pipeline with live UI updates."""
    import curator
    import trainer

    st.session_state.sleep_running = True
    st.session_state.sleep_log = []
    st.session_state.sleep_result = None

    log_container = st.empty()
    progress_bar = st.progress(0, text="Starting Sleep Cycle...")

    def log(msg):
        st.session_state.sleep_log.append(msg)
        log_container.code("\n".join(st.session_state.sleep_log), language="text")

    try:
        # Phase 1: Unload inference model
        log("[sleep] Unloading inference model from memory...")
        unload_model()
        log("[sleep] Model unloaded.")

        # Phase 2: Sanitize
        log("")
        log("=" * 60)
        log("  PHASE 1: SANITIZE (Dataset Generation)")
        log("=" * 60)
        log("")

        def sanitize_progress(idx, total, text):
            if total > 0:
                pct = idx / total
                progress_bar.progress(
                    pct * 0.4,  # Sanitize = 0-40% of progress
                    text=f"Generating QA pairs: {idx}/{total} facts...",
                )
            log(f"[sanitize] Processing fact {idx+1}/{total}: {text[:80]}...")

        sanitize_result = curator.run_sanitize_pipeline(
            n_pairs=4,
            progress_callback=sanitize_progress,
        )

        if sanitize_result["facts_processed"] == 0:
            log("[sanitize] No facts to process!")
            st.session_state.sleep_running = False
            st.session_state.sleep_result = {"success": False}
            return

        log(f"[sanitize] Generated {sanitize_result['qa_generated']} QA pairs")
        log(f"[sanitize] Mixed dataset: {sanitize_result['replay_mixed']} total samples")
        log(f"[sanitize] Train: {sanitize_result['train_count']}, "
            f"Valid: {sanitize_result['valid_count']}")

        progress_bar.progress(0.4, text="Sanitization complete. Starting training...")

        # Phase 3: Sleep (Training)
        log("")
        log("=" * 60)
        log("  PHASE 2: SLEEP (MLX LoRA Training)")
        log("=" * 60)
        log("")

        def train_log(line):
            log(line)
            # Try to parse iteration progress for the progress bar
            if "Iter" in line or "iter" in line:
                progress_bar.progress(
                    min(0.95, 0.4 + 0.55),  # Training = 40-95%
                    text="Training in progress...",
                )

        sleep_result = trainer.run_sleep_cycle(
            fact_ids=sanitize_result["fact_ids"],
            log_callback=train_log,
        )

        progress_bar.progress(1.0, text="Sleep cycle complete!")

        st.session_state.sleep_result = {
            "success": sleep_result["success"],
            "sanitize": sanitize_result,
            "training": sleep_result,
        }

    except Exception as e:
        log(f"[error] {type(e).__name__}: {e}")
        st.session_state.sleep_result = {"success": False, "error": str(e)}

    finally:
        st.session_state.sleep_running = False


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
    st.caption("Sleep/Wake Cycle on Apple M4 — Powered by MLX + Qwen 2.5")

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
