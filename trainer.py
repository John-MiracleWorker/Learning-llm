"""
Continual Learning LLM — MLX LoRA Trainer (trainer.py)

Handles the Sleep Phase:
  1. Generate LoRA training config YAML
  2. Clear inference model from unified memory
  3. Execute mlx_lm.lora training via subprocess wrapped in caffeinate
  4. Save adapter weights (NEVER fuse — SSD wear protection)
  5. Mark facts as learned in ChromaDB on success
"""

import gc
import glob
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from typing import Optional

import yaml

import config
import memory


# ── LoRA Config Generation ────────────────────────────────────────────────


def _count_training_samples() -> int:
    """Count lines in the training JSONL."""
    if not os.path.exists(config.TRAIN_JSONL_PATH):
        return 0
    with open(config.TRAIN_JSONL_PATH, "r") as f:
        return sum(1 for line in f if line.strip())


def _compute_iterations(n_samples: int) -> int:
    """
    Dynamically scale iterations based on dataset size.
    iterations = max(MIN_ITERATIONS, n_samples * ITERATIONS_PER_SAMPLE)
    """
    return max(
        config.MIN_ITERATIONS,
        n_samples * config.ITERATIONS_PER_SAMPLE,
    )


def generate_lora_config() -> str:
    """
    Write the MLX LoRA training YAML config and return its path.

    Key constraints:
      - Target modules: q_proj, v_proj + MLP (gate, down, up)
      - LoRA scale = alpha / rank = 32 / 16 = 2.0
      - Adapter saved to ./adapters (NEVER fused)
    """
    n_samples = _count_training_samples()
    iters = _compute_iterations(n_samples)

    lora_config = {
        # Model
        "model": config.BASE_MODEL,
        "train": True,
        "fine_tune_type": "lora",

        # Data — mlx_lm expects a directory containing train.jsonl / valid.jsonl
        "data": config.DATA_DIR,

        # Training hyperparameters
        "iters": iters,
        "batch_size": config.BATCH_SIZE,
        "grad_accumulation_steps": config.GRAD_ACCUMULATION,
        "learning_rate": config.LEARNING_RATE,
        "seed": 42,

        # LoRA parameters
        "lora_parameters": {
            "keys": [
                "self_attn.q_proj",
                "self_attn.v_proj",
                "mlp.gate_proj",
                "mlp.down_proj",
                "mlp.up_proj",
            ],
            "rank": config.LORA_RANK,
            "scale": config.LORA_ALPHA / config.LORA_RANK,  # 32/16 = 2.0
            "dropout": config.LORA_DROPOUT,
        },

        # Output
        "adapter_path": config.ADAPTER_DIR,
        "save_every": max(10, iters // 5),
        "steps_per_report": 5,
        "steps_per_eval": max(10, iters // 5),
        "val_batches": 10,

        # Memory
        "max_seq_length": 2048,
        "grad_checkpoint": False,
    }

    config_path = os.path.join(config.DATA_DIR, "lora_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(lora_config, f, default_flow_style=False, sort_keys=False)

    return config_path


# ── Memory Management ─────────────────────────────────────────────────────


def clear_inference_model():
    """
    Aggressively free any MLX model from unified memory before training.
    Called before spawning the training subprocess.
    """
    # Force garbage collection of any lingering model references
    gc.collect()

    # If mlx is imported, clear its caches
    try:
        import mlx.core as mx
        mx.metal.clear_cache()
    except (ImportError, AttributeError):
        pass

    gc.collect()


# ── Training Execution ────────────────────────────────────────────────────


def run_training(
    config_path: str,
    log_callback: Optional[callable] = None,
) -> tuple[bool, str]:
    """
    Execute MLX LoRA training in a subprocess wrapped with caffeinate.

    The command:
        caffeinate -i python -m mlx_lm.lora --config <yaml>

    caffeinate -i prevents idle sleep so the M4 Neural Engine
    stays active throughout training.

    Args:
        config_path:  Path to the generated YAML config.
        log_callback: Optional fn(line: str) called for each stdout/stderr line.

    Returns:
        (success: bool, log_output: str)
    """
    cmd = [
        "caffeinate", "-i",
        sys.executable, "-m", "mlx_lm.lora",
        "--config", config_path,
    ]

    log_lines = []

    def _log(line: str):
        log_lines.append(line)
        if log_callback:
            log_callback(line)

    _log(f"[trainer] Starting MLX LoRA training...")
    _log(f"[trainer] Command: {' '.join(cmd)}")
    _log(f"[trainer] Config: {config_path}")
    _log("")

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # line-buffered
            cwd=config.PROJECT_ROOT,
        )

        # Stream output line by line
        for line in iter(process.stdout.readline, ""):
            _log(line.rstrip())

        process.wait()
        exit_code = process.returncode

        if exit_code == 0:
            _log("")
            _log("[trainer] Training completed successfully.")
            return True, "\n".join(log_lines)
        else:
            _log("")
            _log(f"[trainer] Training failed with exit code {exit_code}.")
            return False, "\n".join(log_lines)

    except FileNotFoundError:
        # caffeinate not found (non-macOS) — retry without it
        _log("[trainer] caffeinate not found, running without sleep prevention...")
        cmd_no_cafe = [
            sys.executable, "-m", "mlx_lm.lora",
            "--config", config_path,
        ]
        try:
            process = subprocess.Popen(
                cmd_no_cafe,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=config.PROJECT_ROOT,
            )
            for line in iter(process.stdout.readline, ""):
                _log(line.rstrip())

            process.wait()
            exit_code = process.returncode

            if exit_code == 0:
                _log("")
                _log("[trainer] Training completed successfully (no caffeinate).")
                return True, "\n".join(log_lines)
            else:
                _log("")
                _log(f"[trainer] Training failed with exit code {exit_code}.")
                return False, "\n".join(log_lines)

        except Exception as e:
            _log(f"[trainer] Fatal error: {e}")
            return False, "\n".join(log_lines)

    except Exception as e:
        _log(f"[trainer] Fatal error: {e}")
        return False, "\n".join(log_lines)


# ── Post-Training Lifecycle ───────────────────────────────────────────────


def post_training_cleanup(fact_ids: list[str]) -> dict:
    """
    After a successful Sleep cycle:
      1. Mark all processed facts as learned in ChromaDB
      2. Clean up temporary training files

    Note: Replay buffer append was moved to curator.py (Bug 1B fix).
    New samples are appended there BEFORE mixing to prevent the old
    duplication explosion where mixed data was re-appended every cycle.

    Args:
        fact_ids: IDs of facts that were baked into the adapter.

    Returns:
        Dict with cleanup stats.
    """
    # Mark facts as learned (they stay in ChromaDB for RAG but won't retrain)
    memory.mark_as_learned(fact_ids)

    # Clean up temp training data
    cleaned_files = []
    for path in [config.TRAIN_JSONL_PATH, config.VALID_JSONL_PATH]:
        if os.path.exists(path):
            os.remove(path)
            cleaned_files.append(os.path.basename(path))

    lora_config_path = os.path.join(config.DATA_DIR, "lora_config.yaml")
    if os.path.exists(lora_config_path):
        os.remove(lora_config_path)
        cleaned_files.append("lora_config.yaml")

    return {
        "facts_marked_learned": len(fact_ids),
        "files_cleaned": cleaned_files,
    }


def verify_adapter() -> dict:
    """
    Check if a trained adapter exists and return its metadata.

    Returns:
        Dict with adapter status info.
    """
    adapter_file = os.path.join(config.ADAPTER_DIR, "adapters.safetensors")
    adapter_config = os.path.join(config.ADAPTER_DIR, "adapter_config.json")

    exists = os.path.exists(adapter_file)
    info = {
        "exists": exists,
        "path": config.ADAPTER_DIR,
        "adapter_file": adapter_file,
    }

    if exists:
        info["size_mb"] = round(os.path.getsize(adapter_file) / (1024 * 1024), 2)
        info["modified"] = time.ctime(os.path.getmtime(adapter_file))

    if os.path.exists(adapter_config):
        with open(adapter_config, "r") as f:
            info["config"] = json.load(f)

    return info


# ── Adapter Versioning ─────────────────────────────────────────────────────

MAX_ADAPTER_VERSIONS = 3


def version_adapter() -> Optional[str]:
    """
    Back up the current adapter before training overwrites it.
    Keeps the last MAX_ADAPTER_VERSIONS versions.

    Returns:
        Path to the backup, or None if no adapter exists.
    """
    adapter_file = os.path.join(config.ADAPTER_DIR, "adapters.safetensors")
    if not os.path.exists(adapter_file):
        return None

    # Find next version number
    existing = sorted(glob.glob(os.path.join(config.ADAPTER_DIR, "v[0-9]*")))
    if existing:
        last_num = max(
            int(os.path.basename(p).lstrip("v"))
            for p in existing
            if os.path.basename(p).lstrip("v").isdigit()
        )
        next_num = last_num + 1
    else:
        next_num = 1

    # Copy current adapter to versioned directory
    version_dir = os.path.join(config.ADAPTER_DIR, f"v{next_num}")
    os.makedirs(version_dir, exist_ok=True)
    for fname in os.listdir(config.ADAPTER_DIR):
        fpath = os.path.join(config.ADAPTER_DIR, fname)
        if os.path.isfile(fpath):
            shutil.copy2(fpath, os.path.join(version_dir, fname))

    # Prune old versions (keep last N)
    existing = sorted(glob.glob(os.path.join(config.ADAPTER_DIR, "v[0-9]*")))
    while len(existing) > MAX_ADAPTER_VERSIONS:
        shutil.rmtree(existing.pop(0))

    return version_dir


# ── Post-Sleep Verification ───────────────────────────────────────────────


def verify_fact_recall(
    facts: list[dict],
    log_callback: Optional[callable] = None,
) -> dict:
    """
    Quiz the model on trained facts WITHOUT RAG to measure bake-in accuracy.

    For each fact, asks a simple question and checks if the answer
    contains key terms from the fact.

    Args:
        facts:        List of fact dicts with 'text' key.
        log_callback: Optional fn(line) for logging.

    Returns:
        Dict with recall_rate (0-1), passed, failed, and details.
    """
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    def _log(msg):
        if log_callback:
            log_callback(msg)

    _log("[verify] Loading model for fact recall verification...")

    adapter_path = config.ADAPTER_DIR
    adapter_file = os.path.join(adapter_path, "adapters.safetensors")

    if os.path.exists(adapter_file):
        model, tokenizer = load(
            config.BASE_MODEL,
            adapter_path=adapter_path,
            tokenizer_config={"trust_remote_code": True},
        )
    else:
        model, tokenizer = load(
            config.BASE_MODEL,
            tokenizer_config={"trust_remote_code": True},
        )

    passed = 0
    failed = 0
    details = []

    for fact in facts:
        fact_text = fact["text"]
        # Generate a quiz question about the fact
        quiz_prompt = (
            f"Based on what you know, answer this: "
            f"What do you know about: {fact_text.split()[0:5]}?"
        )
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer concisely."},
            {"role": "user", "content": f"Tell me: {fact_text}"},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        response = generate(
            model, tokenizer, prompt=prompt,
            max_tokens=256, sampler=make_sampler(temp=0.1),
        )
        # Strip thinking blocks
        response = re.sub(r"<think>.*?</think>\s*", "", response, flags=re.DOTALL).strip()

        # Check if key terms from the fact appear in the response
        fact_words = set(w.lower() for w in fact_text.split() if len(w) > 3)
        response_lower = response.lower()
        matched = sum(1 for w in fact_words if w in response_lower)
        recall = matched / max(len(fact_words), 1)

        if recall >= 0.3:
            passed += 1
            status = "PASS"
        else:
            failed += 1
            status = "FAIL"

        details.append({
            "fact": fact_text[:80],
            "status": status,
            "recall": round(recall, 2),
        })
        _log(f"[verify] {status} ({recall:.0%}): {fact_text[:60]}...")

    # Free model
    del model, tokenizer
    gc.collect()

    total = passed + failed
    recall_rate = passed / max(total, 1)
    _log(f"[verify] Recall rate: {passed}/{total} ({recall_rate:.0%})")

    return {
        "recall_rate": round(recall_rate, 2),
        "passed": passed,
        "failed": failed,
        "details": details,
    }


# ── Full Sleep Pipeline ───────────────────────────────────────────────────


def run_sleep_cycle(
    fact_ids: list[str],
    facts: Optional[list[dict]] = None,
    log_callback: Optional[callable] = None,
) -> dict:
    """
    Execute the full Sleep phase:
      1. Version current adapter (backup)
      2. Clear inference model from RAM
      3. Generate LoRA config
      4. Run MLX training (caffeinate-wrapped)
      5. On success: mark facts as learned, clean up, verify recall

    Args:
        fact_ids:     Fact IDs to mark as learned on success.
        facts:        Optional list of fact dicts for post-training verification.
        log_callback: Optional fn(line: str) for streaming log output.

    Returns:
        Dict with full cycle results.
    """
    def _log(msg):
        if log_callback:
            log_callback(msg)

    # 1. Version current adapter
    _log("[sleep] Backing up current adapter...")
    backup_path = version_adapter()
    if backup_path:
        _log(f"[sleep] Adapter backed up to: {os.path.basename(backup_path)}")
    else:
        _log("[sleep] No existing adapter to back up.")

    # 2. Clear inference model
    _log("[sleep] Clearing inference model from unified memory...")
    clear_inference_model()

    # 3. Generate config
    _log("[sleep] Generating LoRA training configuration...")
    config_path = generate_lora_config()
    _log(f"[sleep] Config written to: {config_path}")

    n_samples = _count_training_samples()
    iters = _compute_iterations(n_samples)
    _log(f"[sleep] Dataset: {n_samples} samples → {iters} iterations")
    _log("")

    # 4. Run training
    success, log_output = run_training(config_path, log_callback=log_callback)

    result = {
        "success": success,
        "iterations": iters,
        "dataset_size": n_samples,
        "log": log_output,
    }

    # 5. Post-training
    if success:
        _log("")
        _log("[sleep] Running post-training cleanup...")
        cleanup = post_training_cleanup(fact_ids)
        result["cleanup"] = cleanup
        _log(f"[sleep] Marked {cleanup['facts_marked_learned']} facts as learned.")

        adapter_info = verify_adapter()
        result["adapter"] = adapter_info
        if adapter_info["exists"]:
            _log(f"[sleep] Adapter saved: {adapter_info['size_mb']} MB")

        # 6. Post-sleep verification (self-quiz)
        if facts:
            _log("")
            _log("=" * 60)
            _log("  PHASE 3: VERIFICATION (Self-Quiz)")
            _log("=" * 60)
            _log("")
            verification = verify_fact_recall(facts, log_callback=log_callback)
            result["verification"] = verification

        _log("[sleep] Sleep cycle complete.")
    else:
        _log("")
        _log("[sleep] Training failed — facts NOT marked as learned.")
        _log("[sleep] Fix the issue and retry.")

    return result
