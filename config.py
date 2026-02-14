"""
Continual Learning LLM — Central Configuration
All hyperparameters, paths, and constants in one place.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ADAPTER_DIR = os.path.join(PROJECT_ROOT, "adapters")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CHROMADB_DIR = os.path.join(PROJECT_ROOT, "chromadb_store")

REPLAY_BUFFER_PATH = os.path.join(DATA_DIR, "replay_buffer.jsonl")
TRAIN_JSONL_PATH = os.path.join(DATA_DIR, "train.jsonl")
VALID_JSONL_PATH = os.path.join(DATA_DIR, "valid.jsonl")

# ── Model ──────────────────────────────────────────────────────────────────
BASE_MODEL = "mlx-community/Qwen3-4B-Thinking-2507-4bit"

# ── LoRA Hyperparameters ───────────────────────────────────────────────────
LORA_RANK = 16
LORA_ALPHA = 32          # α = 2r for effective weight shift
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj",
    "v_proj",
    "gate_proj",           # MLP — factual knowledge is localized here
    "down_proj",
    "up_proj",
]

# ── Training ───────────────────────────────────────────────────────────────
LEARNING_RATE = 1e-4
MIN_ITERATIONS = 100
ITERATIONS_PER_SAMPLE = 5  # scale iterations = max(MIN, num_samples * this)
BATCH_SIZE = 1
GRAD_ACCUMULATION = 4      # effective batch size = BATCH_SIZE * GRAD_ACCUMULATION

# ── Replay Mixing Ratio ───────────────────────────────────────────────────
NEW_DATA_RATIO = 0.20      # 20% new facts
OLD_DATA_RATIO = 0.80      # 80% replay buffer
MINIMUM_DATASET_SIZE = 200 # Pad dataset to at least this many samples to prevent overfitting

# ── Fact Conflict Detection ────────────────────────────────────────────────
SUPERSEDE_DISTANCE_THRESHOLD = 0.3  # Cosine distance for LLM-as-Judge candidate retrieval

# ── Memory Compression ────────────────────────────────────────────────────
MEMORY_COMPRESSION_BATCH_SIZE = 20  # Max facts per compression prompt to avoid context blowout

# ── ChromaDB ───────────────────────────────────────────────────────────────
CHROMA_COLLECTION_NAME = "short_term_memory"

# ── RAG ────────────────────────────────────────────────────────────────────
RAG_TOP_K = 5              # number of facts to retrieve per query
RAG_RELEVANCE_THRESHOLD = 0.5  # minimum similarity score (lower = more permissive)

# ── Inference ──────────────────────────────────────────────────────────────
MAX_TOKENS = 1024
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.1
