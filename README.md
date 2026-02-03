# RANKVIDEO: Reasoning Reranking for Text-to-Video Retrieval

Official implementation of **RANKVIDEO**, a video-native reasoning reranker for text-to-video retrieval. RANKVIDEO explicitly reasons over query-video pairs using video content to assess relevance.

🤗 [RankVideo](https://huggingface.co/hltcoe/RankVideo/tree/main)

## Overview

RANKVIDEO is trained using a two-stage curriculum:
1. **Perception-grounded SFT**: The model learns to generate captions grounded in video content
2. **Reranking Training**: Fine-tuning with pointwise, pairwise, and teacher distillation objectives

Given a query-video pair, RANKVIDEO predicts relevance by comparing log-probabilities of discrete answer tokens, producing a scalar relevance score.



## Installation

```bash
git clone https://github.com/tskow99/RANKVIDEO-Reasoning-Reranker.git
cd RANKVIDEO-Reasoning-Reranker
```

### Create Conda Environment

```bash
conda create -n RankVideo python=3.12 pip -y
conda activate RankVideo
```

### Install Dependencies

Below is the minimum set up dependencies that should allow for running inference code.

```bash
# Upgrade pip and core tools
python -m pip install -U pip setuptools wheel

# Install ffmpeg via conda
conda install -c conda-forge -y ffmpeg

# Install PyTorch with CUDA 12.8 support
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128

# Install vLLM for efficient inference
pip install vllm==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu128

# Install remaining dependencies
pip install transformers==4.57.1
pip install "qwen-vl-utils[decord]==0.0.14"
pip install jsonlines==4.0.0 tqdm==4.67.1
pip install pytrec_eval-terrier
pip install trl
pip install peft
pip install bitsandbytes
```

## Configuration

Before running, update the paths in `config.py`:

```python
# Model paths
BASE_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
HF_MODEL_PATH = "hltcoe/RankVideo"
CACHE_DIR = "/path/to/model/cache"

# Data paths
TRAIN_DATA_PATH = "/path/to/train_data.jsonl"
EVAL_DATA_PATH = "/path/to/eval_data.jsonl"
VIDEO_DIR = "/path/to/videos"
```

## Pretrained Weights

Download pretrained [RANKVIDEO weights](https://huggingface.co/hltcoe/RankVideo/tree/main) from HuggingFace:

```python
from rankvideo import VLMReranker
reranker = VLMReranker(
    model_path="hltcoe/RankVideo",
    cache_dir="/path/to/cache"
)
```

## Usage

### Inference

Score query-video pairs for relevance:

```python
from rankvideo import VLMReranker

reranker = VLMReranker(model_path="hltcoe/RankVideo")

scores = reranker.score_batch(
    queries=["example query 1", "example query2"],
    video_paths=["/path/to/video1.mp4", "/path/to/video2.mp4"],
)

for score in scores:
    print(f"P(relevant) = {score['p_yes']:.3f}")
    print(f"Logit delta = {score['logit_delta']:.3f}")
```

### Reranking First-Stage Results

```bash
python -m rankvideo.rerank \
    --model ORG/rankvideo-reranker \
    --video2queries data/video2queries.json \
    --query-mapping data/queries.tsv \
    --video-dir /path/to/videos \
    --output-dir outputs/reranking
```

### Evaluation

```bash
python -m rankvideo.evaluate \
    --pred-dir outputs/reranking \
    --first-stage data/first_stage_results.json \
    --qrels data/qrels.txt
```

## Data Synthesis Pipeline

RANKVIDEO includes scripts for generating training data:

### Generate Video Captions

Generate perception-grounded captions for videos using a VLM:

```bash
python -m rankvideo.generate_captions \
    --video-root /path/to/videos \
    --output data/captions.jsonl \
    --model-name Qwen/Qwen3-Omni-30B-A3B-Instruct
```

### Generate Teacher Reasoning Labels

Generate reasoning traces and soft labels from a teacher model for distillation:

```bash
python -m rankvideo.generate_reasoning \
    --queries data/queries.tsv \
    --candidates data/candidates.trec \
    --captions-jsonl data/captions.jsonl \
    --output data/teacher_labels.jsonl \
    --model-path /path/to/reasoning/model \
    --topk 20
```

## Training

### Stage 1: Perception-Grounded SFT

```bash
python -m rankvideo.train_perception \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --train-data /path/to/caption_data.jsonl \
    --eval-data /path/to/caption_eval.jsonl \
    --output-dir outputs/stage1
```

### Stage 2: Reranker Training

```bash
python -m rankvideo.train_reranker \
    --model outputs/stage1/perception-TIMESTAMP \
    --train-data /path/to/ranking_data.jsonl \
    --eval-data /path/to/ranking_eval.jsonl \
    --output-dir outputs/stage2
```

## Data Format

### Training Data (JSONL)

Each line should be a JSON object with:

```json
{
  "query_id": "q001",
  "query": "person playing guitar on stage",
  "doc_id": "video_123",
  "videos": ["/path/to/video_123.mp4"],
  "true_label": 1,
  "teacher_p_yes": 0.85,
  "evidence": {
    "caption": "A musician performs with an acoustic guitar...",
    "asr": "transcribed speech if available"
  }
}
```

### Query Mapping (TSV)

```
query_id	query_text
q001	person playing guitar on stage
q002	dog running in park
```

### First-Stage Results (JSON)

```json
{
  "q001": {"video_123": 0.95, "video_456": 0.82, ...},
  "q002": {"video_789": 0.91, ...}
}
```

### TREC Qrels

```
q001 0 video_123 1
q001 0 video_456 0
```

## Provided Data Files

We provide preprocessed data files to reproduce our results. Extract the archive:

```bash
tar -xzvf data.tar.gz
```

This creates the following structure:

```
data/
├── training_data.json                           # Training data with teacher labels
├── videos2queriesranking_AV_OmniEmbed.json      # Video-to-query candidate mapping
└── first_stage_results/
    └── ranking_AV_OmniEmbed.json                # First-stage retrieval scores
```

### File Descriptions

#### `training_data.json`
Training examples with teacher reasoning traces for distillation. Each line contains:
- `query_id`, `doc_id`: Identifiers for the query-video pair
- `query`: The text query
- `evidence`: Contains `caption` (video description) and `asr` (speech transcript)
- `teacher_reasoning`: Reasoning trace from the teacher model

**Used for**: Stage 2 reranker training (`--train-data`)

#### `videos2queriesranking_AV_OmniEmbed.json`
Maps each video ID to the list of query IDs it is a candidate for:
```json
{"video_id": ["query_1", "query_2", ...], ...}
```

**Used for**: Batch reranking (`--video2queries` in `rerank.py`)

#### `first_stage_results/ranking_AV_OmniEmbed.json`
First-stage retrieval scores from OmniEmbed. Maps query IDs to candidate videos with scores:
```json
{"query_id": {"video_id": score, ...}, ...}
```

**Used for**: Evaluation baseline and cascade reranking (`--first-stage` in `evaluate.py`)

### Reproducing Results

1. **Download MultiVENT 2.0 videos** from [multivent.github.io](https://multivent.github.io/)

2. **Run reranking** on the first-stage candidates:
```bash
python -m rankvideo.rerank \
    --model hltcoe/RankVideo \
    --video2queries data/videos2queriesranking_AV_OmniEmbed.json \
    --video-dir /path/to/multivent/videos \
    --output-dir outputs/reranking
```

3. **Evaluate** against the first-stage baseline:
```bash
python -m rankvideo.evaluate \
    --pred-dir outputs/reranking \
    --first-stage data/first_stage_results/ranking_AV_OmniEmbed.json \
    --qrels /path/to/multivent/qrels.txt
```

## Dataset

For training from scratch, we use the [MultiVENT 2.0](https://multivent.github.io/) dataset, a large-scale multilingual video retrieval benchmark.

## Citation
# to be updated once published
```bibtex
@misc{skow2026rankvideoreasoningrerankingtexttovideo,
      title={RANKVIDEO: Reasoning Reranking for Text-to-Video Retrieval}, 
      author={Tyler Skow and Alexander Martin and Benjamin Van Durme and Rama Chellappa and Reno Kriz},
      year={2026},
      eprint={2602.02444},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2602.02444}, 
}
```

## License

This project is released under the MIT License.
