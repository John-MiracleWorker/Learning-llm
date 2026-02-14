"""
Continual Learning LLM — Background Training Worker (worker.py)

Decouples the Sleep Cycle pipeline from the Streamlit UI thread (Bug 3B).

Instead of blocking the Streamlit WebSocket for 10+ minutes during MLX
training, the UI writes a start_training.flag file and this worker
picks it up, runs the full pipeline, and writes progress to status.json.

Usage:
    # Start the worker alongside Streamlit:
    python worker.py &
    streamlit run app.py

    # Or use the combined launcher (recommended):
    # The worker auto-exits when idle for >1 hour.
"""

from __future__ import annotations

import gc
import json
import os
import signal
import sys
import time
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import config

# ── Paths ─────────────────────────────────────────────────────────────────

FLAG_PATH = os.path.join(config.DATA_DIR, "start_training.flag")
STATUS_PATH = os.path.join(config.DATA_DIR, "training_status.json")
POLL_INTERVAL = 2  # seconds between flag checks
IDLE_TIMEOUT = 3600  # auto-exit after 1 hour idle


def write_status(status: dict):
    """Atomically write status JSON (write to tmp then rename)."""
    os.makedirs(config.DATA_DIR, exist_ok=True)
    tmp_path = STATUS_PATH + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(status, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, STATUS_PATH)


def read_flag() -> dict | None:
    """Read and parse the training flag file, return None if not found."""
    if not os.path.exists(FLAG_PATH):
        return None
    try:
        with open(FLAG_PATH, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def clear_flag():
    """Remove the flag file after processing."""
    if os.path.exists(FLAG_PATH):
        os.remove(FLAG_PATH)


def run_pipeline(flag_params: dict):
    """
    Execute the full Sanitize → Dream (optional) → Train pipeline.
    Writes progress updates to status.json throughout.
    """
    import curator
    import trainer

    dream_enabled = flag_params.get("dream_enabled", False)
    log_lines = []

    def log(msg):
        log_lines.append(msg)
        write_status({
            "state": "running",
            "phase": current_phase,
            "started_at": started_at,
            "log": log_lines[-50:],  # Keep last 50 lines to avoid huge JSON
            "updated_at": datetime.now().isoformat(),
        })

    started_at = datetime.now().isoformat()
    current_phase = "sanitize"

    try:
        # Phase 1: Sanitize
        log("[worker] Starting Sanitize phase...")
        write_status({
            "state": "running",
            "phase": "sanitize",
            "started_at": started_at,
            "log": log_lines,
            "updated_at": datetime.now().isoformat(),
        })

        def sanitize_progress(idx, total, text):
            log(f"[sanitize] Processing fact {idx+1}/{total}: {text[:80]}...")

        sanitize_result = curator.run_sanitize_pipeline(
            n_pairs=4,
            progress_callback=sanitize_progress,
        )

        if sanitize_result["facts_processed"] == 0:
            log("[sanitize] No facts to process!")
            write_status({
                "state": "completed",
                "success": False,
                "reason": "no_facts",
                "started_at": started_at,
                "finished_at": datetime.now().isoformat(),
                "log": log_lines,
            })
            return

        log(f"[sanitize] Generated {sanitize_result['qa_generated']} QA pairs")
        log(f"[sanitize] Mixed dataset: {sanitize_result['replay_mixed']} total samples")
        log(f"[sanitize] Train: {sanitize_result['train_count']}, "
            f"Valid: {sanitize_result['valid_count']}")

        # Phase 2: Dream (optional) — runs BEFORE training (Bug 1A fix)
        dream_result = None
        if dream_enabled:
            current_phase = "dream"
            log("")
            log("[worker] Starting Dream phase...")
            import dreams
            dream_result = dreams.run_dream_enhanced_sleep(
                fact_ids=sanitize_result["fact_ids"],
                facts=sanitize_result.get("facts"),
                log_callback=log,
            )
            gc.collect()

        # Phase 3: Train
        current_phase = "training"
        log("")
        log("[worker] Starting Training phase...")

        sleep_result = trainer.run_sleep_cycle(
            fact_ids=sanitize_result["fact_ids"],
            facts=sanitize_result.get("facts"),
            log_callback=log,
        )

        # Write final status
        write_status({
            "state": "completed",
            "success": sleep_result["success"],
            "started_at": started_at,
            "finished_at": datetime.now().isoformat(),
            "sanitize": {
                "facts_processed": sanitize_result["facts_processed"],
                "qa_generated": sanitize_result["qa_generated"],
                "train_count": sanitize_result["train_count"],
            },
            "training": {
                "iterations": sleep_result.get("iterations", 0),
                "dataset_size": sleep_result.get("dataset_size", 0),
                "verification": sleep_result.get("verification"),
            },
            "dreams": dream_result,
            "log": log_lines[-100:],
        })

        log("[worker] Pipeline complete!")

    except Exception as e:
        log(f"[error] {type(e).__name__}: {e}")
        write_status({
            "state": "failed",
            "error": str(e),
            "error_type": type(e).__name__,
            "started_at": started_at,
            "finished_at": datetime.now().isoformat(),
            "log": log_lines[-100:],
        })


def main():
    """Main worker loop — watches for flag, runs pipeline, repeats."""
    print(f"[worker] Background training worker started (PID {os.getpid()})")
    print(f"[worker] Watching for: {FLAG_PATH}")
    print(f"[worker] Status output: {STATUS_PATH}")

    # Handle graceful shutdown
    def signal_handler(sig, frame):
        print("\n[worker] Shutting down gracefully...")
        write_status({"state": "worker_stopped", "stopped_at": datetime.now().isoformat()})
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    idle_since = time.time()

    while True:
        flag = read_flag()
        if flag is not None:
            print(f"[worker] Flag detected! Starting pipeline...")
            clear_flag()
            idle_since = time.time()

            run_pipeline(flag)

            # Force garbage collection after pipeline
            gc.collect()
            print(f"[worker] Pipeline finished. Resuming watch...")
        else:
            # Check idle timeout
            if time.time() - idle_since > IDLE_TIMEOUT:
                print(f"[worker] Idle timeout ({IDLE_TIMEOUT}s). Exiting.")
                break

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
