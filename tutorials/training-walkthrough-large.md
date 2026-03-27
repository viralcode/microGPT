# How We Trained superGPT Large (350M) — Full Walkthrough

**A real-world, start-to-finish account of training a 421M parameter GPT model on 1 billion tokens of educational text using a single cloud GPU. This tutorial covers everything: hardware selection, data preparation, training configuration, monitoring, debugging production issues, and testing the final model.**

---

## Table of Contents

1. [Overview](#1-overview)
2. [Hardware & Cost](#2-hardware--cost)
3. [Model Architecture](#3-model-architecture)
4. [Dataset: FineWeb-Edu](#4-dataset-fineweb-edu)
5. [Data Preparation Pipeline](#5-data-preparation-pipeline)
6. [Training Configuration](#6-training-configuration)
7. [Launching Training](#7-launching-training)
8. [Monitoring Progress](#8-monitoring-progress)
9. [Production Bugs We Hit (And Fixed)](#9-production-bugs-we-hit-and-fixed)
10. [Final Results](#10-final-results)
11. [Text Generation Tests](#11-text-generation-tests)
12. [Downloading the Model](#12-downloading-the-model)
13. [Lessons Learned](#13-lessons-learned)

---

## 1. Overview

We trained a **421M parameter GPT-style transformer** from scratch using:

- **1 billion tokens** from the FineWeb-Edu dataset (educational web text)
- A single **NVIDIA A40 (46GB VRAM)** rented on RunPod
- **~14 hours** of total training time (including restarts from debugging)
- **Total cost: ~$5.75** at $0.41/hour

The model learned to generate fluent, grammatically correct English paragraphs in an educational writing style.

---

## 2. Hardware & Cost

We rented a single A40 GPU on [RunPod](https://runpod.io):

| Resource | Spec |
|----------|------|
| GPU | NVIDIA A40 (46GB VRAM) |
| CPU | 16 vCPUs |
| RAM | 503 GB system memory |
| Disk | 20GB network volume (`/workspace/`) |
| Cost | $0.41/hour |
| Template | `RunPod Pytorch 2.4.0` |

### Why the A40?

The A40 is a "sweet spot" GPU for training models in the 100M–1B parameter range:
- 46GB VRAM fits models up to ~1B params comfortably
- BF16 support for efficient mixed-precision training
- $0.41/hr is affordable for multi-hour training runs
- Unlike consumer GPUs (3090, 4090), it doesn't thermal throttle in data center environments

---

## 3. Model Architecture

superGPT uses a modern GPT-4 style transformer with architectural improvements over the original GPT-2:

```
Preset: large
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Layers:           24
Heads:            16 (query)
KV Heads:         4  (grouped-query attention)
Embedding Dim:    1024
Block Size:       2048 (context window)
Vocab Size:       151,643 (Qwen tokenizer)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Parameters: 421.1M
```

### Key Architecture Features

| Feature | What It Does | Why It Matters |
|---------|-------------|----------------|
| **GQA (Grouped Query Attention)** | 16 query heads share 4 KV heads | 4× less KV-cache memory, faster inference |
| **RoPE (Rotary Position Embeddings)** | Injects position info via rotation matrices | Better long-range pattern recognition than learned embeddings |
| **SwiGLU Activation** | Gated activation in feed-forward layers | ~1-2% better loss vs standard GELU (used in LLaMA, PaLM) |
| **RMSNorm** | Simplified layer normalization | 10-15% faster than LayerNorm with similar quality |

### Parameter Count Breakdown

The 421M total breaks down roughly as:
- **Embedding layers**: ~155M (vocab_size × embed_dim)
- **Attention layers**: ~100M (24 layers × QKV projections)
- **Feed-forward layers**: ~160M (24 layers × SwiGLU MLP)
- **Output head**: ~6M (tied with embeddings in some configs)

---

## 4. Dataset: FineWeb-Edu

We used [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), a curated subset of web crawl data filtered for educational quality.

### Why FineWeb-Edu?

| Property | Value |
|----------|-------|
| Total size | ~1.3 trillion tokens |
| We used | 1 billion tokens (~0.08%) |
| Content type | Educational articles, tutorials, textbooks |
| Language | English |
| Quality filter | Scored by a classifier trained to identify educational content |

FineWeb-Edu produces significantly better models than raw Common Crawl data because:
1. **Higher signal-to-noise ratio** — no spam, ads, or boilerplate
2. **Structured writing** — articles with clear intros, explanations, and conclusions
3. **Rich vocabulary** — covers science, history, mathematics, technology, etc.
4. **Consistent grammar** — editorial-quality text teaches the model proper English

---

## 5. Data Preparation Pipeline

Before training, we tokenized the raw text into binary shards that PyTorch can load efficiently.

### Step 1: Tokenization

We used the **Qwen/Qwen2.5-0.5B** tokenizer (BPE, vocab size 151,643):

```bash
python -m supergpt.training.data_pipeline \
    --dataset HuggingFaceFW/fineweb-edu \
    --tokenizer Qwen/Qwen2.5-0.5B \
    --output-dir data \
    --num-tokens 1000000000 \
    --shard-size 10000000
```

This downloads text from HuggingFace in streaming mode (no full download needed), tokenizes it into integer sequences, and writes binary `.bin` shard files.

### Step 2: Merging Shards

The pipeline then merges all shards into two memory-mapped files:

```
data/
├── train.bin          # Training data (~950M tokens, 90%)
├── val.bin            # Validation data (~50M tokens, 10%)
├── meta.pkl           # Metadata (vocab_size, tokenizer_type)
└── shard_000.bin ...  # Individual shards (105 files, 7.5GB total)
```

### Why Memory-Mapped Binary Files?

Instead of loading text and tokenizing on-the-fly during training (which would bottleneck the GPU), we pre-tokenize everything into raw `uint32` arrays stored on disk. During training, PyTorch uses `numpy.memmap` to read random chunks directly from the file without loading the entire dataset into RAM. This means:

- **Zero CPU bottleneck** — the GPU never waits for data
- **Constant memory usage** — regardless of dataset size
- **Random access** — any batch can be read in microseconds

---

## 6. Training Configuration

### Hyperparameters

```python
preset          = "large"          # 421M params
batch_size      = 8                # 8 sequences per step
block_size      = 2048             # 2048 tokens per sequence
tokens_per_step = 16,384           # batch_size × block_size
learning_rate   = 1.5e-4           # Peak LR
lr_schedule     = "cosine"         # Cosine annealing with warmup
warmup_iters    = 200              # Linear warmup for first 200 steps
max_iters       = 30,000           # Total training iterations
weight_decay    = 0.1              # AdamW regularization
grad_clip       = 1.0              # Gradient clipping
dtype           = "bfloat16"       # Mixed precision training
compile         = True             # torch.compile() optimization
```

### Why These Values?

**Batch size 8**: The A40 has 46GB VRAM. With a 421M param model, BF16 weights + optimizer states + activations at batch_size=8 use ~42GB — close to the limit but safe.

**Learning rate 1.5e-4**: This follows the Chinchilla scaling law. For a 421M model, the optimal LR is roughly in the 1e-4 to 3e-4 range. We chose 1.5e-4 as a conservative middle ground.

**Cosine LR schedule**: Starts at 0, warms up to 1.5e-4 over 200 steps, then smoothly decays back to near-zero by iteration 30,000. This gives the model a "gentle landing" that squeezes out the last bits of quality.

**30,000 iterations**: With 16,384 tokens per step, this processes ~491M tokens. The Chinchilla-optimal ratio suggests training tokens should be ~20× the parameter count (421M × 20 = 8.4B), so we're actually undertrained. More iterations would improve quality further.

**torch.compile()**: PyTorch 2.0's JIT compiler fuses operations and generates optimized CUDA kernels. On the A40, this gave us ~20,800 tokens/sec vs ~15,000 without compilation — a **39% speedup** at the cost of a one-time 5-10 minute compilation delay before training starts.

### VRAM Usage Breakdown

```
Model weights (BF16):    ~  0.8 GB
Optimizer states (FP32): ~ 3.2 GB   (Adam keeps 2 copies)
Gradients (BF16):        ~  0.8 GB
Activations:             ~ 35.0 GB  (batch_size × layers × attention maps)
PyTorch overhead:        ~  2.0 GB
─────────────────────────────────────
Total:                   ~ 42.0 GB  (of 46 GB available)
```

---

## 7. Launching Training

### The Launch Script

We used a `nohup` background process to ensure training survives SSH disconnections:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nohup python -u -m supergpt.training.train \
    --preset large \
    --data-dir data \
    --max-iters 30000 \
    --batch-size 8 \
    --lr 1.5e-4 \
    --device cuda \
    --lr-schedule cosine \
    --compile \
    > /workspace/bigrun.log 2>&1 &
```

Key flags explained:
- **`nohup ... &`**: Runs in background, survives SSH disconnect
- **`python -u`**: Unbuffered output so logs appear in real-time
- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`**: Prevents VRAM fragmentation
- **`--compile`**: Enables torch.compile() for 39% speed boost
- **`> /workspace/bigrun.log 2>&1`**: All output goes to a log file for monitoring

### What Happens After Launch

1. **Model initialization** (~5 seconds): Builds the 421M parameter model on GPU
2. **torch.compile()** (~5-10 minutes): Compiles the entire model into optimized CUDA kernels. GPU shows 100% utilization during this phase with no log output — this is normal!
3. **Step 0**: First evaluation runs, prints initial loss (~12.0 for random weights)
4. **Training loop**: Processes ~20,800 tokens/sec, printing every 100 iterations

---

## 8. Monitoring Progress

### Monitor Script

We built a custom monitoring script (`scripts/monitor.sh`) that SSHs into the pod and provides a live dashboard:

```
╔══════════════════════════════════════════════════════════╗
║     ⚡ superGPT RunPod Training Monitor ⚡              ║
║     Model: large (350M) | Data: 1B FineWeb-Edu         ║
╚══════════════════════════════════════════════════════════╝

  🏋️  PHASE 2: TRAINING (350M params)

  [████████████████████████████████████████] 100%

  Iteration: 30,000 / 30,000
  Loss:      3.60
  Val Loss:  3.5834
  Speed:     20,800 tok/s
  ETA:       0h 0m remaining

  ── GPU ──
  Util:      100%  |  Memory: 42053/46068 MiB
  Temp:      70°C  |  Power: 295W
```

Usage:
```bash
./scripts/monitor.sh
```

### Understanding the Loss Curve

The training loss tells you how well the model is learning:

| Loss | Interpretation |
|------|---------------|
| ~12.0 | Random model (iteration 0) — guessing uniformly from 151K tokens |
| ~8.0 | Learning basic token frequencies |
| ~6.0 | Learning common words and simple patterns |
| ~5.0 | Learning grammar and sentence structure |
| ~4.0 | Learning coherent multi-sentence writing |
| ~3.5 | Strong paragraph-level coherence (our final result) |
| ~3.0 | Would need 10×+ more data and compute |

### Key Metrics to Watch

- **Train loss should steadily decrease** — if it plateaus early, LR may be too low
- **Val loss should track train loss** — if val loss rises while train drops, you're overfitting
- **Speed (tok/s) should be stable** — sudden drops indicate thermal throttling or I/O bottleneck
- **GPU temp should stay under 85°C** — data center GPUs are fine up to 83°C

---

## 9. Production Bugs We Hit (And Fixed)

Real-world training is never smooth. Here are the four production bugs we encountered and how we solved them:

### Bug 1: Checkpoint Corruption on Network Filesystem

**Symptom**: Training ran 30K iterations successfully, but `best.pt` only contained weights from iteration 500.

**Root Cause**: RunPod's `/workspace/` is a network-mounted filesystem (MFS). PyTorch saves checkpoints as ZIP files using `torch.save()`. When writing 5GB+ files to the network drive, the atomic `shutil.move()` operation silently failed with:
```
Warning: Checkpoint save failed: [enforce fail at inline_container.cc:603]
unexpected pos 1071962944 vs 1071962832
```

**Fix**: Modified `CheckpointManager._atomic_save()` to write temp files to `/tmp/` (local SSD) first, then copy to the network volume:
```python
# Before (broken on network FS):
fd, tmp_path = tempfile.mkstemp(dir=self.checkpoint_dir, suffix=".pt.tmp")

# After (writes to local SSD first):
fd, tmp_path = tempfile.mkstemp(dir="/tmp", suffix=".pt.tmp")
```

**Lesson**: Never assume the filesystem your code runs on is a local disk. Cloud providers often use network-attached storage with different consistency guarantees.

---

### Bug 2: Disk Quota Exceeded

**Symptom**: Training crashed after 2,000 iterations. RunPod dashboard showed Container Disk at 100%.

**Root Cause**: The checkpoint manager was saving periodic numbered backups (`step_2000.pt`, `step_4000.pt`, etc.), each 5.1GB. Combined with the dataset (7.5GB) and two active checkpoints (10.2GB), the total exceeded RunPod's 20GB workspace volume quota:

```
Dataset:       7.5 GB
latest.pt:     5.1 GB
best.pt:       5.1 GB
step_2000.pt:  5.1 GB  ← This pushed us over!
─────────────────────
Total:        22.8 GB  > 20 GB quota
```

**Fix**: Disabled numbered step checkpoints entirely. Only `latest.pt` and `best.pt` are saved:
```python
# Removed:
if iter_num > 0 and iter_num % 2000 == 0:
    shutil.copy2(target_path, f"step_{iter_num}.pt")
```

**Lesson**: Always calculate your total disk budget before training. `model_size × num_checkpoint_copies + dataset_size` must fit within your storage quota.

---

### Bug 3: CUDA Out of Memory on Resume

**Symptom**: `torch.OutOfMemoryError: Tried to allocate 4.63 GiB` when resuming training from a checkpoint.

**Root Cause**: `torch.load(resume_path, map_location=device)` was loading the entire 5.1GB checkpoint dictionary directly into GPU VRAM. This temporarily doubled the model's memory footprint (model weights + checkpoint dictionary both on GPU), exceeding the A40's 46GB limit right when `torch.compile()` needed additional memory.

**Fix**: Load checkpoint to CPU memory (500GB available) instead, then explicitly free it:
```python
# Before (loads 5GB into VRAM):
checkpoint = torch.load(resume_path, map_location=device)

# After (loads 5GB into system RAM):
checkpoint = torch.load(resume_path, map_location="cpu")
raw_model.load_state_dict(checkpoint["model"])
optimizer.load_state_dict(checkpoint["optimizer"])
del checkpoint
gc.collect()
torch.cuda.empty_cache()
```

**Lesson**: When your VRAM is nearly full, always load checkpoints to CPU. The `del + gc.collect() + empty_cache()` pattern is essential for preventing memory leaks during long training runs.

---

### Bug 4: Monitor Script Showing Wrong ETA and Val Loss

**Symptom**: Monitor showed "ETA: 0h 36m" (should have been 5+ hours) and "Val Loss: 7000" (should have been 4.08).

**Root Cause**: Two bugs in `monitor.sh`:
1. `TOKS_PER_ITER` was hardcoded to 2048 (block_size only) instead of 16384 (batch_size × block_size), making the ETA 8× too fast.
2. The val loss regex was grabbing the first number from the log line (`Step 7000 | val loss: 4.08`), which matched "7000" instead of "4.08".

**Fix**: 
```bash
# ETA fix:
TOKS_PER_ITER=16384  # was 2048

# Val loss regex fix:
VAL_LOSS=$(echo "$LOGS" | grep "val" | tail -1 | \
    grep -oE 'val loss:? *[0-9.]+' | grep -oE '[0-9.]+')
```

---

## 10. Final Results

### Training Metrics

| Metric | Value |
|--------|-------|
| Final train loss | 3.5478 |
| **Best val loss** | **3.5834** |
| Best checkpoint | Iteration 29,500 |
| Total iterations | 30,000 |
| Tokens processed | ~491M |
| Peak speed | 20,900 tok/s |
| Average speed | 20,700 tok/s |

### Loss Progression

```
Iter     0: loss 12.12  (random weights)
Iter  1000: loss  5.44  (learning word frequencies)
Iter  3000: loss  4.65  (learning grammar)
Iter  5000: loss  4.30  (learning sentence structure)
Iter 10000: loss  4.00  (coherent paragraphs)
Iter 20000: loss  3.70  (fluent educational writing)
Iter 30000: loss  3.58  (final — strong coherence)
```

---

## 11. Text Generation Tests

After training, we tested the model with three prompts:

### Test 1: "Artificial intelligence is"

> Artificial intelligence is a set of skills which are used to identify, recognize, and respond to human emotions in a way that is easier to understand. A team of researchers at the University of Oxford's Department of Human Behavior have been able to map the cognitive abilities of people with different emotional states and to assess how their ability to respond to their emotions with a better understanding of human emotions...

**Verdict**: ✅ Fluent English, coherent topic, proper sentence structure. Facts are hallucinated but plausible-sounding.

### Test 2: "The history of science began when"

> The history of science began when it was first published in the 18th century. The first scientific articles were published in 1803 and 1804. The British Academy was founded in 1815 by the British Academy. It was formed in 1815...

**Verdict**: ✅ Good paragraph flow and educational tone. Dates are fabricated and it gets repetitive ("British Academy" loop), which is typical for models at this scale.

### Test 3: "Learning to code is important because"

> Learning to code is important because it is so easy to put a code to work, and you don't have to be able to put it to code. You can use a number of different tools to learn the code that you want to code, including: An online programming language called XML, A program that has an open source program...

**Verdict**: ✅ Understood the educational prompt format and produced a bullet-pointed list. Content is nonsensical but the structure and grammar are correct.

### Generation Settings

```bash
python scripts/generate.py \
    --checkpoint trained_models/large_350m_fineweb/best.pt \
    --prompt "Your prompt here" \
    --max-tokens 200 \
    --temperature 0.7
```

- **Temperature 0.7**: Good balance between creativity and coherence
- **Top-k 50**: Limits sampling to the 50 most likely next tokens
- **KV-Cache**: Enabled for faster autoregressive generation (~53 tokens/sec)

---

## 12. Downloading the Model

After training completes, pull the model to your local machine:

```bash
# Create local directory
mkdir -p trained_models/large_350m_fineweb

# Download checkpoint (5.1GB, takes ~3 min at 30MB/s)
scp -P 22020 root@<RUNPOD_IP>:/workspace/superGPT/checkpoints/best.pt \
    trained_models/large_350m_fineweb/best.pt

# Download tokenizer metadata
scp -P 22020 root@<RUNPOD_IP>:/workspace/superGPT/data/meta.pkl \
    trained_models/large_350m_fineweb/meta.pkl
```

Then **stop your RunPod pod** immediately to stop billing!

---

## 13. Lessons Learned

### What Worked Well
1. **torch.compile()** gave a free 39% speed boost with zero code changes
2. **BF16 mixed precision** halved memory usage with no quality loss
3. **FineWeb-Edu** — high-quality data matters more than data quantity
4. **Cosine LR schedule** — smooth decay produced steadily declining loss with no instabilities
5. **nohup background training** — survived SSH disconnections and monitoring from local machine

### What We'd Do Differently
1. **Use local SSD for checkpoints from the start** — would have avoided the corruption bug entirely
2. **Calculate disk budget upfront** — `dataset + (num_checkpoints × model_size)` before launching
3. **Load checkpoints to CPU by default** — simple one-line change prevents OOM on resume
4. **Train for longer** — 30K iterations processed only 491M tokens. Chinchilla scaling suggests 8.4B tokens would be optimal for a 421M model. More training = better output quality
5. **Use a larger batch size with gradient accumulation** — would have improved training stability

### Scaling Beyond This

To train a significantly better model, here's what you'd change:

| Goal | Change | Expected Impact |
|------|--------|----------------|
| Better coherence | Train on 5-10B tokens | Major quality improvement |
| Longer context | Increase block_size to 4096 | Better long-range understanding |
| Stronger model | Use `xl` preset (1B params) | Needs H100 or multi-A40 |
| Faster training | Multi-GPU with FSDP | Linear speedup with GPU count |
| Less hallucination | Fine-tune on instruction data | See `instruction-tuning-chat.md` |

---

*Tutorial by the superGPT team. Model trained March 26-27, 2026.*
