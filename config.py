"""
Configuration for RANKVIDEO training and inference.

Update the paths below before running any scripts.
"""

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Base VLM model for training (Qwen3-VL recommended)
BASE_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

# Path to Stage 1 checkpoint (perception-grounded model)
# After running train_perception.py, set this to the output directory
STAGE1_CHECKPOINT = "/path/to/stage1/checkpoint"

# HuggingFace model hub path for pretrained RANKVIDEO weights
# Download from: https://huggingface.co/YOUR_ORG/rankvideo-reranker
HF_MODEL_PATH = "YOUR_ORG/rankvideo-reranker"

# Local cache directory for model weights
CACHE_DIR = "/path/to/model/cache"

# =============================================================================
# DATA PATHS
# =============================================================================

# Training data (JSONL format with messages, videos, query_id, doc_id fields)
TRAIN_DATA_PATH = "/path/to/train_data.jsonl"
EVAL_DATA_PATH = "/path/to/eval_data.jsonl"

# Video directory root (videos organized in shards or flat)
VIDEO_DIR = "/path/to/videos"

# Query mapping file (TSV: query_id \t query_text)
QUERY_MAPPING_PATH = "/path/to/queries.tsv"

# First stage retrieval results (JSON: {query_id: {doc_id: score, ...}, ...})
FIRST_STAGE_RESULTS = "/path/to/first_stage_results.json"

# TREC qrels file for evaluation
QRELS_PATH = "/path/to/qrels.txt"

# =============================================================================
# OUTPUT PATHS  
# =============================================================================

OUTPUT_DIR = "/path/to/output"

# =============================================================================
# VIDEO PROCESSING
# =============================================================================

FPS = 2.0
MAX_FRAMES = 32
NFRAMES = 64

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================

SEED = 42
USE_QLORA = True
LEARNING_RATE = 1e-5
NUM_EPOCHS = 2
WARMUP_RATIO = 0.03
GRADIENT_ACCUMULATION_STEPS = 16

# LoRA configuration
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "up_proj", "down_proj", "gate_proj",
    "mm_projector", "vision_proj", "visual_projection",
    "resampler", "image_projection", "multi_modal_projector",
]

# Stage 2 loss hyperparameters
TAU_RANK = 10.0
TAU_DISTILL = 5.0
TAU_POINT = 1.0
LAMBDA_DISTILL = 5.0
LAMBDA_POINT = 1.0

# =============================================================================
# WANDB (optional)
# =============================================================================

LOG_TO_WANDB = False
WANDB_PROJECT = "rankvideo"
WANDB_API_KEY = ""  # Set via environment variable WANDB_API_KEY preferred
