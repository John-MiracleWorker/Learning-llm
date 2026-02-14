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
import json
import math
import os
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


# ── Full Sleep Pipeline ───────────────────────────────────────────────────


def run_sleep_cycle(
    fact_ids: list[str],
    log_callback: Optional[callable] = None,
) -> dict:
    """
    Execute the full Sleep phase:
      1. Clear inference model from RAM
      2. Generate LoRA config
      3. Run MLX training (caffeinate-wrapped)
      4. On success: mark facts as learned, clean up

    Args:
        fact_ids:     Fact IDs to mark as learned on success.
        log_callback: Optional fn(line: str) for streaming log output.

    Returns:
        Dict with full cycle results.
    """
    def _log(msg):
        if log_callback:
            log_callback(msg)

    # 1. Clear inference model
    _log("[sleep] Clearing inference model from unified memory...")
    clear_inference_model()

    # 2. Generate config
    _log("[sleep] Generating LoRA training configuration...")
    config_path = generate_lora_config()
    _log(f"[sleep] Config written to: {config_path}")

    n_samples = _count_training_samples()
    iters = _compute_iterations(n_samples)
    _log(f"[sleep] Dataset: {n_samples} samples → {iters} iterations")
    _log("")

    # 3. Run training
    success, log_output = run_training(config_path, log_callback=log_callback)

    result = {
        "success": success,
        "iterations": iters,
        "dataset_size": n_samples,
        "log": log_output,
    }

    # 4. Post-training
    if success:
        _log("")
        _log("[sleep] Running post-training cleanup...")
        cleanup = post_training_cleanup(fact_ids)
        result["cleanup"] = cleanup

        adapter_info = verify_adapter()
        result["adapter"] = adapter_info
        if adapter_info["exists"]:
            _log(f"[sleep] Adapter saved: {adapter_info['size_mb']} MB")
        _log("[sleep] Sleep cycle complete.")
    else:
        _log("")
        _log("[sleep] Training failed — facts NOT marked as learned.")
        _log("[sleep] Fix the issue and retry.")

    return result
