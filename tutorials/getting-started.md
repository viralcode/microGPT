# Getting Started with superGPT

**A complete guide to training, fine-tuning, and deploying your own GPT models using superGPT.**

---

## Table of Contents

1. [What is superGPT?](#1-what-is-supergpt)
2. [Installation](#2-installation)
3. [Architecture Overview](#3-architecture-overview)
4. [Quick Start: Train Your First Model](#4-quick-start-train-your-first-model)
5. [Understanding Model Presets](#5-understanding-model-presets)
6. [Data Preparation](#6-data-preparation)
7. [Training Deep Dive](#7-training-deep-dive)
8. [Text Generation](#8-text-generation)
9. [Fine-Tuning with LoRA](#9-fine-tuning-with-lora)
10. [Knowledge Distillation](#10-knowledge-distillation)
11. [Multi-GPU Training (FSDP)](#11-multi-gpu-training-fsdp)
12. [Advanced Features](#12-advanced-features)
13. [Monitoring & Debugging](#13-monitoring--debugging)
14. [Exporting & Serving](#14-exporting--serving)
15. [Troubleshooting](#15-troubleshooting)

---

## 1. What is superGPT?

superGPT is a production-grade LLM training framework that implements the latest architectural innovations from leading labs:

| Feature | Description | From |
|---------|-------------|------|
| **Multi-Head Attention (MHA)** | Standard transformer attention | GPT-2/3 |
| **Grouped Query Attention (GQA)** | Reduced KV heads for efficiency | Llama 2 |
| **Multi-head Latent Attention (MLA)** | Low-rank KV compression | DeepSeek-V3 |
| **Mixture of Experts (MoE)** | Sparse expert routing | DeepSeek-V3 |
| **Multi-Token Prediction (MTP)** | Predict N future tokens simultaneously | DeepSeek-V3 |
| **Rotary Position Embeddings (RoPE)** | Relative position encoding | LLaMA |
| **SwiGLU Activation** | Gated linear unit with Swish | LLaMA |
| **Sliding Window Attention** | Fixed window for efficiency | Mistral |
| **Native Sparse Attention (NSA)** | Learned sparse patterns | DeepSeek (2025) |
| **FP8 Training** | 8-bit floating point on H100 | DeepSeek-V3 |
| **LoRA Fine-Tuning** | Low-rank adapter training | Microsoft |
| **Speculative Decoding** | Fast inference with draft model | Google |
| **FSDP** | Multi-GPU model parallelism | PyTorch |

### Project Structure

```
supergpt/
├── core/
│   ├── config.py          # All model & training configurations
│   ├── model.py           # GPT model (MHA, GQA, MLA, MoE, etc.)
│   └── flash_mla.py       # FlashAttention for MLA
├── training/
│   ├── train.py           # Main training loop (single & multi-GPU)
│   ├── data_pipeline.py   # Streaming data pipeline + quality filtering
│   ├── finetune.py        # LoRA fine-tuning
│   ├── distill.py         # Knowledge distillation
│   ├── lora.py            # LoRA module implementation
│   ├── fp8_utils.py       # FP8 training utilities (H100)
│   ├── parallel.py        # FSDP parallelism helpers
│   ├── streaming.py       # Streaming dataset utilities
│   └── expert_parallel.py # MoE expert parallelism
├── inference/
│   ├── generate.py        # Text generation with sampling
│   ├── evaluate.py        # Model evaluation & benchmarks
│   ├── export.py          # Export to ONNX/TorchScript
│   └── serve.py           # HTTP API server
├── alignment/
│   ├── align.py           # DPO alignment
│   ├── rlhf.py            # RLHF with PPO
│   └── rlvr.py            # RLVR (rule-based verification)
└── tools/
    └── visualize.py       # Architecture visualization dashboard
```

---

## 2. Installation

### Prerequisites

- Python 3.10+
- PyTorch 2.0+ (with CUDA for GPU training)
- A GPU with at least 8GB VRAM (for training; CPU works for inference)

### Setup

```bash
# Clone the repository
git clone https://github.com/viralcode/superGPT.git
cd superGPT

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install torch torchvision torchaudio
pip install transformers datasets tiktoken numpy

# Optional: for advanced features
pip install wandb          # Training monitoring
pip install tensorboard    # Training visualization
pip install flash-attn     # Flash Attention (faster training)
pip install onnx           # Model export
```

### Verify Installation

```bash
python -c "import supergpt; print('superGPT loaded successfully!')"
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

---

## 3. Architecture Overview

### The Transformer Block

Every superGPT model is built from stacked transformer blocks:

```
Input Tokens
     │
     ▼
┌─────────────────┐
│  Token Embedding │ ← Convert token IDs to vectors
│  + Position Enc  │ ← RoPE (Rotary Positional Embedding)
└────────┬────────┘
         │
         ▼  (repeat N times)
┌─────────────────────────────┐
│    Layer Norm               │
│         │                   │
│    Attention                │
│    ┌────┴─────┐             │
│    │MHA / GQA │ ← Standard  │
│    │  MLA     │ ← DeepSeek  │
│    │  NSA     │ ← Sparse    │
│    └────┬─────┘             │
│         │ + residual        │
│    Layer Norm               │
│         │                   │
│    Feed Forward             │
│    ┌────┴─────┐             │
│    │ SwiGLU   │ ← Dense     │
│    │ MoE      │ ← Sparse    │
│    └────┬─────┘             │
│         │ + residual        │
└─────────┴───────────────────┘
         │
         ▼
┌─────────────────┐
│   Layer Norm    │
│   LM Head       │ ← Project to vocab size
│   (+ MTP Heads) │ ← Multi-Token Prediction
└─────────────────┘
```

### Attention Variants

| Variant | KV Heads | Memory | Speed | Best For |
|---------|----------|--------|-------|----------|
| **MHA** | Same as Q | High | Baseline | Small models (≤1B) |
| **GQA** | 1/4 of Q | ~60% | ~1.3× | Medium models (1-10B) |
| **MLA** | Latent | ~20% | ~1.5× | Large models (≥10B) |

```python
# MHA: 6 query heads, 6 KV heads (full attention)
GPTConfig(n_head=6, n_kv_head=6)

# GQA: 12 query heads, 4 KV heads (grouped)
GPTConfig(n_head=12, n_kv_head=4)

# MLA: Low-rank latent attention (DeepSeek V3)
GPTConfig(use_mla=True, kv_lora_rank=512, q_lora_rank=1536)
```

---

## 4. Quick Start: Train Your First Model

The fastest way to get started — train on Shakespeare in under 5 minutes:

```bash
# Step 1: Prepare Shakespeare data
python -m supergpt.training.data_pipeline --dataset shakespeare --output-dir data/

# Step 2: Train (CPU is fine for this test)
python -m supergpt.training.train \
    --preset small \
    --data-dir data/ \
    --max-iters 1000 \
    --device cpu

# Step 3: Generate text
python -m supergpt.inference.generate \
    --checkpoint checkpoints/best.pt \
    --prompt "To be or not to be" \
    --max-tokens 200
```

**With a GPU** (much faster — 2 minutes):

```bash
python -m supergpt.training.train \
    --preset small \
    --data-dir data/ \
    --max-iters 2000 \
    --compile \
    --device cuda
```

---

## 5. Understanding Model Presets

superGPT comes with pre-configured model sizes:

| Preset | Layers | Heads | Embed | Params | Context | GPU VRAM | Training Time |
|--------|--------|-------|-------|--------|---------|----------|---------------|
| `small` | 6 | 6 | 384 | ~10M | 256 | <2 GB | Minutes |
| `medium` | 12 | 12 | 768 | ~124M | 512 | ~8 GB | Hours |
| `large` | 24 | 16 | 1024 | ~350M | 2048 | ~24 GB | Hours |
| `xl` | 24 | 16 | 2048 | ~774M | 4096 | ~40 GB | Days |
| `gpt4` | 32 | 32 | 4096 | ~1.3B | 8192 | ~80 GB | Days |
| `deepseek` | 27 | 16 | 2048 | ~680M | 4096 | ~40 GB | Days |
| `mistral` | 32 | 32 | 4096 | ~1.3B | 8192 | ~80 GB | Days |
| `gemma2` | 28 | 16 | 2304 | ~500M | 8192 | ~40 GB | Days |

### Custom Configuration

```bash
# Override any preset parameter
python -m supergpt.training.train \
    --preset medium \
    --n-layer 8 \
    --n-head 8 \
    --block-size 1024 \
    --data-dir data/
```

### DeepSeek V3 Architecture

```bash
# Train with the full DeepSeek V3 feature set:
# MLA + MoE (64 experts, 6 active) + MTP (3 tokens) + shared experts
python -m supergpt.training.train \
    --preset deepseek \
    --data-dir data/ \
    --max-iters 50000 \
    --compile
```

---

## 6. Data Preparation

### Option A: Built-In Datasets

```bash
# Shakespeare (tiny, for testing)
python -m supergpt.training.data_pipeline --dataset shakespeare --output-dir data/

# FineWeb-Edu (high-quality web text, any size)
python -m supergpt.training.data_pipeline \
    --dataset HuggingFaceFW/fineweb-edu \
    --tokenizer Qwen/Qwen2.5-0.5B \
    --max-tokens 100000000 \
    --output-dir data/

# Wikipedia
python -m supergpt.training.data_pipeline \
    --dataset wikipedia \
    --subset 20220301.en \
    --tokenizer Qwen/Qwen2.5-0.5B \
    --max-tokens 500000000 \
    --output-dir data/
```

### Option B: Any HuggingFace Dataset

```bash
# The data pipeline works with ANY HuggingFace dataset that has a "text" column
python -m supergpt.training.data_pipeline \
    --dataset allenai/dolma \
    --tokenizer Qwen/Qwen2.5-0.5B \
    --max-tokens 1000000000 \
    --output-dir data/ \
    --text-field text

# Custom text field name
python -m supergpt.training.data_pipeline \
    --dataset bigcode/starcoderdata \
    --tokenizer Qwen/Qwen2.5-0.5B \
    --text-field content \
    --output-dir data/
```

### Option C: Your Own Text Files

```bash
# Put .txt files in a directory, then tokenize them:
mkdir -p my_data/
# Copy your text files into my_data/
python -m supergpt.training.data_pipeline \
    --input-dir my_data/ \
    --tokenizer Qwen/Qwen2.5-0.5B \
    --output-dir data/
```

### Quality Filtering & Deduplication

The data pipeline includes built-in quality filtering:

```bash
# With quality filtering + dedup (default)
python -m supergpt.training.data_pipeline \
    --dataset HuggingFaceFW/fineweb-edu \
    --output-dir data/

# Disable filtering (raw data)
python -m supergpt.training.data_pipeline \
    --dataset HuggingFaceFW/fineweb-edu \
    --no-quality-filter \
    --no-dedup \
    --output-dir data/
```

**Quality filters applied:**
- Minimum 50 characters, 10 words
- Maximum 100K characters
- ASCII ratio ≥ 90% (English filter)
- Special character ratio ≤ 10%
- Bigram repetition ≤ 30%
- MinHash deduplication (Jaccard similarity > 0.8)

### What Gets Created

```
data/
├── train.bin    # Tokenized training data (uint32 binary)
├── val.bin      # Tokenized validation data
└── meta.pkl     # Metadata (vocab size, tokenizer, stats)
```

---

## 7. Training Deep Dive

### Basic Training

```bash
python -m supergpt.training.train \
    --preset small \
    --data-dir data/ \
    --max-iters 10000 \
    --batch-size 32 \
    --lr 3e-4 \
    --device cuda
```

### Key Training Parameters

| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| Model preset | `--preset` | `small` | Model architecture |
| Data directory | `--data-dir` | `data` | Path to tokenized data |
| Max iterations | `--max-iters` | 5000 | Total training steps |
| Batch size | `--batch-size` | 64 | Sequences per batch |
| Learning rate | `--lr` | 3e-4 | Peak learning rate |
| LR schedule | `--lr-schedule` | `cosine` | `cosine` or `wsd` |
| Warmup | `--warmup-iters` | 100 | LR warmup steps |
| Grad accumulation | `--grad-accum` | 1 | Effective batch multiplier |
| Eval interval | `--eval-interval` | 500 | Evaluate every N steps |
| Gradient clipping | `--grad-clip` | 1.0 | Max gradient norm |
| Weight decay | `--weight-decay` | 0.1 | AdamW weight decay |
| Compile | `--compile` | off | `torch.compile()` (2× speed) |
| Device | `--device` | `auto` | `cuda`, `cpu`, `mps` |
| Dtype | `--dtype` | `auto` | `bfloat16`, `float16`, `float32` |

### Using torch.compile (Highly Recommended)

```bash
# ~2× faster training on GPU
python -m supergpt.training.train \
    --preset small \
    --data-dir data/ \
    --compile \
    --device cuda
```

> **Note:** First iteration takes 1-2 minutes to compile. After that, training is ~2× faster.

### Gradient Accumulation (Simulate Larger Batches)

```bash
# Effective batch size = batch_size × grad_accum = 8 × 16 = 128
python -m supergpt.training.train \
    --preset large \
    --batch-size 8 \
    --grad-accum 16 \
    --device cuda
```

### Auto-Resume

superGPT automatically resumes from the latest checkpoint:

```bash
# First run: trains from scratch, writes checkpoints/latest.pt
python -m supergpt.training.train --preset small --max-iters 5000

# If interrupted, just re-run — it auto-resumes from iter 5000:
python -m supergpt.training.train --preset small --max-iters 10000
```

---

## 8. Text Generation

### Basic Generation

```bash
python -m supergpt.inference.generate \
    --checkpoint checkpoints/best.pt \
    --prompt "The theory of general relativity states that" \
    --max-tokens 200 \
    --temperature 0.8
```

### Sampling Parameters

| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| Temperature | `--temperature` | 0.8 | Randomness (0=greedy, 1=creative) |
| Top-k | `--top-k` | 50 | Keep top K tokens |
| Top-p | `--top-p` | 0.9 | Nucleus sampling threshold |
| Min-p | `--min-p` | 0.05 | Minimum probability threshold |
| Repetition penalty | `--rep-penalty` | 1.1 | Penalize repeated tokens |
| Max tokens | `--max-tokens` | 200 | Max tokens to generate |

### Interactive Mode

```bash
# Chat-like interface
python -m supergpt.inference.generate \
    --checkpoint checkpoints/best.pt \
    --interactive

# You:> What is machine learning?
# Model: Machine learning is a branch of artificial intelligence...
```

### Speculative Decoding (2-3× Faster)

```bash
# Use a small "draft" model to speed up generation
python -m supergpt.inference.generate \
    --checkpoint checkpoints/large_model.pt \
    --draft-checkpoint checkpoints/small_model.pt \
    --spec-k 5 \
    --prompt "Explain quantum computing"
```

---

## 9. Fine-Tuning with LoRA

LoRA (Low-Rank Adaptation) lets you fine-tune with minimal memory:

```bash
# Fine-tune on your custom data
python -m supergpt.training.finetune \
    --checkpoint checkpoints/best.pt \
    --data-dir data/my_finetune_data/ \
    --lora-rank 64 \
    --lora-alpha 16 \
    --max-iters 5000 \
    --lr 2e-5 \
    --batch-size 16 \
    --compile --device cuda
```

### LoRA Parameters

| Rank | Memory Savings | Quality | Best For |
|------|---------------|---------|----------|
| 4-8 | ~95% | Good | Quick experiments |
| 16-32 | ~90% | Great | Domain adaptation |
| 64-128 | ~80% | Excellent | Full fine-tuning substitute |

---

## 10. Knowledge Distillation

Transfer knowledge from a large "teacher" model to a smaller "student":

```bash
python -m supergpt.training.distill \
    --teacher checkpoints/large_model.pt \
    --student-preset small \
    --data-dir data/ \
    --alpha 0.7 \
    --temperature 2.0 \
    --max-iters 10000 \
    --compile --device cuda
```

- **alpha**: Balance between teacher soft targets (higher α) and hard targets (lower α)
- **temperature**: Higher = softer teacher output = more knowledge transfer

---

## 11. Multi-GPU Training (FSDP)

Scale to multiple GPUs with Fully Sharded Data Parallel:

```bash
# 4 GPUs
torchrun --nproc_per_node=4 \
    -m supergpt.training.train \
    --preset large \
    --data-dir data/ \
    --distributed \
    --compile

# 8 GPUs across 2 nodes
torchrun --nproc_per_node=4 --nnodes=2 --node-rank=0 \
    --master-addr=node0 --master-port=29500 \
    -m supergpt.training.train \
    --preset xl \
    --distributed \
    --compile
```

---

## 12. Advanced Features

### Mixture of Experts (MoE)

```bash
# Train with 8 experts, 2 active per token
python -m supergpt.training.train \
    --preset small \
    --use-moe \
    --n-experts 8 \
    --n-experts-active 2 \
    --data-dir data/ \
    --compile --device cuda
```

### Multi-Token Prediction (MTP)

```bash
# Predict 3 tokens ahead simultaneously (DeepSeek V3 style)
python -m supergpt.training.train \
    --preset small \
    --n-predict-tokens 3 \
    --data-dir data/ \
    --compile --device cuda
```

### FP8 Training (H100 Only)

```bash
# 2× memory savings on H100 GPUs
python -m supergpt.training.train \
    --preset xl \
    --fp8 \
    --data-dir data/ \
    --compile --device cuda
```

### Sliding Window Attention (Mistral Style)

```bash
python -m supergpt.training.train \
    --preset small \
    --sliding-window 1024 \
    --block-size 4096 \
    --data-dir data/
```

---

## 13. Monitoring & Debugging

### WandB Integration

```bash
pip install wandb
wandb login

python -m supergpt.training.train \
    --preset small \
    --data-dir data/ \
    --wandb
```

### TensorBoard

```bash
pip install tensorboard

python -m supergpt.training.train \
    --preset small \
    --data-dir data/ \
    --tensorboard

# View in browser:
tensorboard --logdir runs/
```

### Architecture Visualizer

```bash
# Interactive dashboard showing model architecture, attention patterns, weights
python -m supergpt.tools.visualize --checkpoint checkpoints/best.pt
# Opens browser at http://localhost:5000
```

---

## 14. Exporting & Serving

### Export to ONNX

```bash
python -m supergpt.inference.export \
    --checkpoint checkpoints/best.pt \
    --format onnx \
    --output model.onnx
```

### HTTP API Server

```bash
python -m supergpt.inference.serve \
    --checkpoint checkpoints/best.pt \
    --port 8000

# Test:
curl "http://localhost:8000/generate?prompt=Hello%20world&max_tokens=100"
```

---

## 15. Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| `CUDA out of memory` | Reduce `--batch-size`, add `--grad-accum`, or use a smaller `--preset` |
| `torch.compile() slow` | First iteration takes 1-2 min. Add `--compile` flag (worth the wait) |
| `Loss not decreasing` | Check data quality, try lower `--lr`, ensure data isn't corrupted |
| `Generated text is garbage` | Train longer, check tokenizer matches between train and generate |
| `uint16/uint32 mismatch` | Auto-detected now! Ensure `meta.pkl` has correct `vocab_size` |
| `Checkpoint size mismatch` | Delete old checkpoints if switching presets: `rm checkpoints/*.pt` |
| `NaN in loss` | Reduce `--lr`, add `--grad-clip 1.0`, check for data corruption |
| `Multi-GPU hangs` | Ensure `NCCL_DEBUG=INFO`, check `--nproc_per_node` matches GPU count |

### Getting Help

- **GitHub Issues**: [github.com/viralcode/superGPT/issues](https://github.com/viralcode/superGPT/issues)
- **Training Data Guide**: [tutorials/training-data-guide.md](training-data-guide.md)
- **RunPod Deployment**: [tutorials/deploy-runpod.md](deploy-runpod.md)
