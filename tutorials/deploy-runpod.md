# Deploy superGPT on RunPod for GPU Training

**A step-by-step guide to renting cloud GPUs on RunPod and training LLMs with superGPT — from zero to a trained model.**

---

## Table of Contents

1. [Why RunPod?](#1-why-runpod)
2. [Creating a RunPod Account](#2-creating-a-runpod-account)
3. [Choosing Your GPU](#3-choosing-your-gpu)
4. [Launching a Pod](#4-launching-a-pod)
5. [Setting Up superGPT](#5-setting-up-supergpt)
6. [Preparing Training Data](#6-preparing-training-data)
7. [Running Training](#7-running-training)
8. [Monitoring Training](#8-monitoring-training)
9. [Testing Your Model](#9-testing-your-model)
10. [Downloading Your Model](#10-downloading-your-model)
11. [Multi-GPU Training](#11-multi-gpu-training)
12. [Automation Script](#12-automation-script)
13. [Cost Optimization Tips](#13-cost-optimization-tips)
14. [Troubleshooting](#14-troubleshooting)

---

## 1. Why RunPod?

RunPod provides on-demand cloud GPUs at competitive prices. Here's how it compares:

| Provider | A100 80GB | H100 80GB | Min Billing | Spot Pricing |
|----------|-----------|-----------|-------------|-------------|
| **RunPod** | ~$1.64/hr | ~$3.29/hr | Per second | Yes (40% off) |
| AWS SageMaker | ~$4.10/hr | ~$8.22/hr | Per hour | Yes |
| Google Cloud | ~$3.67/hr | ~$11.98/hr | Per minute | Yes |
| Lambda Labs | ~$1.10/hr | ~$2.49/hr | Per hour | No |

RunPod advantages:
- **Per-second billing** — only pay for what you use
- **Spot instances** — 40-70% cheaper for fault-tolerant training
- **Pre-built templates** — PyTorch, CUDA, cuDNN ready to go
- **Persistent storage** — your data survives pod restarts
- **SSH access** — full control, just like a local machine

---

## 2. Creating a RunPod Account

1. Go to [runpod.io](https://www.runpod.io/)
2. Click **Sign Up** → create account with email or Google
3. Go to **Billing** → add a payment method
4. Add credits ($10-25 is enough for testing)
5. Go to **Settings** → **SSH Keys** → add your public SSH key:

```bash
# On your local machine:
# If you don't have an SSH key yet:
ssh-keygen -t ed25519 -C "your@email.com"

# Copy your public key:
cat ~/.ssh/id_ed25519.pub
# → Paste this into RunPod's SSH Keys settings
```

---

## 3. Choosing Your GPU

### GPU Selection Guide

| Your Goal | GPU | VRAM | RunPod Cost | What You Can Train |
|-----------|-----|------|-------------|-------------------|
| **Testing / Learning** | RTX 4090 | 24 GB | ~$0.44/hr | Models up to ~350M params |
| **Serious Training** | A100 80GB | 80 GB | ~$1.64/hr | Models up to ~1.5B params |
| **Large Models** | H100 80GB | 80 GB | ~$3.29/hr | Models up to ~3B params + FP8 |
| **Multi-GPU** | 4× A100 | 320 GB | ~$6.56/hr | Models up to ~7B params |
| **Production** | 8× H100 | 640 GB | ~$26.32/hr | Models up to ~70B params |

### Model Size → GPU Requirements

| superGPT Preset | Params | Min VRAM | Recommended GPU |
|----------------|--------|----------|-----------------|
| `small` | ~10M | 2 GB | Any (even CPU) |
| `medium` | ~124M | 8 GB | RTX 4090 / A100 |
| `large` | ~350M | 24 GB | A100 40GB |
| `xl` | ~774M | 40 GB | A100 80GB |
| `gpt4` | ~1.3B | 60 GB | A100 80GB |
| `deepseek` | ~680M | 40 GB | A100 80GB |

> **Pro tip:** Start with an RTX 4090 ($0.44/hr) for testing, then scale up to A100 for real training.

---

## 4. Launching a Pod

### Step-by-Step

1. Go to [runpod.io/console/pods](https://www.runpod.io/console/pods)
2. Click **+ Deploy**
3. Choose your GPU (e.g., **1x A100 80GB**)
4. Select template: **RunPod PyTorch 2.1** (or latest PyTorch template)
5. Set **Volume Disk**: 50-100 GB (for data + checkpoints)
6. Set **Container Disk**: 20 GB
7. Click **Deploy On-Demand** (or **Spot** for cheaper)

### Connect via SSH

After the pod starts (1-2 minutes):

```bash
# Find your SSH command in the RunPod dashboard:
# Pod → Connect → SSH Terminal
# It will look like:
ssh root@<IP_ADDRESS> -p <PORT> -i ~/.ssh/id_ed25519

# Example:
ssh root@194.68.245.39 -p 22193 -i ~/.ssh/id_ed25519
```

### Verify GPU

```bash
# On the RunPod machine:
nvidia-smi
# Should show your GPU with full VRAM available

python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
# → CUDA: True, GPU: NVIDIA A100 80GB
```

---

## 5. Setting Up superGPT

Once you're SSH'd into RunPod:

```bash
# Navigate to persistent storage (survives restarts)
cd /workspace

# Clone superGPT
git clone https://github.com/viralcode/superGPT.git
cd superGPT

# Install dependencies (PyTorch is already installed on RunPod)
pip install transformers datasets tiktoken numpy

# Optional: for monitoring
pip install wandb

# Verify everything works
python -c "import supergpt; print('superGPT ready!')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}, Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.0f} GB')"
```

---

## 6. Preparing Training Data

### Option A: FineWeb-Edu (Recommended for Quality)

```bash
# 100M tokens (~15 min to prepare on RunPod)
python -m supergpt.training.data_pipeline \
    --dataset HuggingFaceFW/fineweb-edu \
    --tokenizer Qwen/Qwen2.5-0.5B \
    --max-tokens 100000000 \
    --output-dir data/

# 1B tokens (~2 hours to prepare)
python -m supergpt.training.data_pipeline \
    --dataset HuggingFaceFW/fineweb-edu \
    --tokenizer Qwen/Qwen2.5-0.5B \
    --max-tokens 1000000000 \
    --output-dir data/
```

### Option B: Quick Test with Shakespeare

```bash
python -m supergpt.training.data_pipeline \
    --dataset shakespeare \
    --output-dir data/
```

### Option C: Custom Dataset from HuggingFace

```bash
# Train on code
python -m supergpt.training.data_pipeline \
    --dataset bigcode/starcoderdata \
    --tokenizer Qwen/Qwen2.5-0.5B \
    --text-field content \
    --max-tokens 500000000 \
    --output-dir data/

# Train on Wikipedia
python -m supergpt.training.data_pipeline \
    --dataset wikipedia \
    --subset 20220301.en \
    --tokenizer Qwen/Qwen2.5-0.5B \
    --max-tokens 500000000 \
    --output-dir data/
```

### Verify Data

```bash
ls -lh data/
# train.bin  ~380 MB (100M tokens × 4 bytes)
# val.bin    ~7.8 MB
# meta.pkl   ~1 KB

python -c "
import numpy as np, pickle
train = np.memmap('data/train.bin', dtype=np.uint32, mode='r')
with open('data/meta.pkl','rb') as f: meta = pickle.load(f)
print(f'Tokens: {len(train):,}')
print(f'Vocab:  {meta[\"vocab_size\"]:,}')
print(f'Size:   {train.nbytes / 1e9:.2f} GB')
"
```

---

## 7. Running Training

### Basic Training (Single GPU)

```bash
# Train a small model (10M params, ~1 hour)
python -m supergpt.training.train \
    --preset small \
    --data-dir data/ \
    --max-iters 10000 \
    --batch-size 32 \
    --lr 3e-4 \
    --compile \
    --device cuda
```

### Medium Model (124M params)

```bash
# Requires ~8GB VRAM (A100 or H100)
python -m supergpt.training.train \
    --preset medium \
    --data-dir data/ \
    --max-iters 50000 \
    --batch-size 16 \
    --lr 1.5e-4 \
    --compile \
    --device cuda
```

### Background Training (Keeps Running After SSH Disconnect)

```bash
# IMPORTANT: Use nohup so training continues even if SSH disconnects
nohup python -u -m supergpt.training.train \
    --preset small \
    --data-dir data/ \
    --max-iters 10000 \
    --batch-size 32 \
    --lr 3e-4 \
    --compile \
    --device cuda \
    > /workspace/training.log 2>&1 &

echo "Training started! PID: $!"
echo "Monitor with: tail -f /workspace/training.log"
```

> **Critical:** Always use `nohup` for training runs. SSH connections can drop at any time. Without nohup, your training will be killed when SSH disconnects.

### Full Training Script (One Command)

```bash
#!/bin/bash
# save as: /workspace/run_training.sh

set -e

echo "============================================"
echo "  superGPT Training Pipeline"
echo "============================================"

cd /workspace/superGPT

# Step 1: Prepare data (skip if already done)
if [ ! -f data/train.bin ]; then
    echo "Step 1: Preparing data..."
    python -m supergpt.training.data_pipeline \
        --dataset HuggingFaceFW/fineweb-edu \
        --tokenizer Qwen/Qwen2.5-0.5B \
        --max-tokens 100000000 \
        --output-dir data/
else
    echo "Step 1: Data already prepared, skipping."
fi

# Step 2: Train
echo "Step 2: Starting training..."
python -u -m supergpt.training.train \
    --preset small \
    --data-dir data/ \
    --max-iters 10000 \
    --batch-size 32 \
    --lr 3e-4 \
    --compile \
    --device cuda

# Step 3: Test generation
echo "Step 3: Testing generation..."
python -m supergpt.inference.generate \
    --checkpoint checkpoints/best.pt \
    --prompt "The most important concepts in machine learning are" \
    --max-tokens 200 \
    --temperature 0.7

echo "============================================"
echo "  Training complete!"
echo "  Checkpoint: checkpoints/best.pt"
echo "============================================"
```

Run it:
```bash
chmod +x /workspace/run_training.sh
nohup /workspace/run_training.sh > /workspace/full_run.log 2>&1 &
tail -f /workspace/full_run.log
```

---

## 8. Monitoring Training

### From Your Local Machine

```bash
# Check training progress
ssh root@<IP> -p <PORT> -i ~/.ssh/id_ed25519 "tail -20 /workspace/training.log"

# Check GPU utilization
ssh root@<IP> -p <PORT> -i ~/.ssh/id_ed25519 "nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv"

# Check if training is still running
ssh root@<IP> -p <PORT> -i ~/.ssh/id_ed25519 "ps aux | grep train | grep -v grep"
```

### What to Look For

```
  Step      0 | train loss: 11.9940 | val loss: 11.9936 | lr: 0.00e+00   ← Starting (random)
  iter    100 | loss 8.0898 | 61411 tok/s | lr 3.00e-04                   ← Dropping fast (good!)
  iter    500 | loss 6.1243 | 61372 tok/s | lr 2.99e-04                   ← Still dropping
  Step   1000 | train loss: 5.5000 | val loss: 5.6000 | lr: 2.95e-04     ← Val > Train = slightly overfitting (ok)
  Step   5000 | train loss: 4.8000 | val loss: 5.0000 | lr: 1.50e-04     ← Converging
  Step  10000 | train loss: 4.5000 | val loss: 4.7000 | lr: 3.00e-05     ← Final (cosine decay)
```

**Good signs:**
- Loss decreasing consistently
- Validation loss close to training loss
- GPU utilization > 80%
- 50K+ tokens/second throughput

**Bad signs:**
- Loss stuck or increasing → lower learning rate
- Val loss much higher than train loss → overfitting, need more data
- GPU utilization < 50% → increase batch size
- NaN in loss → reduce learning rate, check data

### With WandB (Pretty Dashboard)

```bash
pip install wandb
wandb login  # Enter your API key

python -m supergpt.training.train \
    --preset small \
    --data-dir data/ \
    --wandb \
    --compile --device cuda
```

---

## 9. Testing Your Model

After training completes:

```bash
cd /workspace/superGPT

# Generate text
python -m supergpt.inference.generate \
    --checkpoint checkpoints/best.pt \
    --prompt "In the field of artificial intelligence, transformers are" \
    --max-tokens 300 \
    --temperature 0.7

# Try different prompts
python -m supergpt.inference.generate \
    --checkpoint checkpoints/best.pt \
    --prompt "def fibonacci(n):" \
    --max-tokens 200 \
    --temperature 0.3

# Interactive mode
python -m supergpt.inference.generate \
    --checkpoint checkpoints/best.pt \
    --interactive
```

---

## 10. Downloading Your Model

### Option A: SCP (Simplest)

```bash
# From your LOCAL machine:
scp -P <PORT> -i ~/.ssh/id_ed25519 \
    root@<IP>:/workspace/superGPT/checkpoints/best.pt \
    ./my_model.pt

# Download everything
scp -r -P <PORT> -i ~/.ssh/id_ed25519 \
    root@<IP>:/workspace/superGPT/checkpoints/ \
    ./checkpoints/
```

### Option B: Push to GitHub

```bash
# On RunPod:
cd /workspace/superGPT
git add checkpoints/best.pt
git commit -m "Add trained model checkpoint"
git push

# Then pull on your local machine
```

### Option C: Upload to HuggingFace Hub

```bash
pip install huggingface_hub
huggingface-cli login

python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj='checkpoints/best.pt',
    path_in_repo='model.pt',
    repo_id='your-username/my-supergpt-model',
    repo_type='model',
)
print('Uploaded to HuggingFace!')
"
```

---

## 11. Multi-GPU Training

### RunPod Multi-GPU Pods

1. In RunPod, select a multi-GPU option (e.g., **4× A100 80GB**)
2. SSH in and verify:

```bash
nvidia-smi
# Should show 4 GPUs

python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
# → GPUs: 4
```

3. Run with FSDP:

```bash
cd /workspace/superGPT

torchrun --nproc_per_node=4 \
    -m supergpt.training.train \
    --preset large \
    --data-dir data/ \
    --max-iters 50000 \
    --distributed \
    --compile
```

### Multi-GPU Performance

| GPUs | Preset | Params | Throughput | Time for 10K iters |
|------|--------|--------|-----------|---------------------|
| 1× A100 | small | 10M | ~60K tok/s | ~45 min |
| 1× A100 | medium | 124M | ~30K tok/s | ~2 hrs |
| 4× A100 | large | 350M | ~80K tok/s | ~3 hrs |
| 8× A100 | xl | 774M | ~100K tok/s | ~5 hrs |

---

## 12. Automation Script

Save this on your **local machine** to automate the entire process:

```bash
#!/bin/bash
# File: train_on_runpod.sh
# Usage: ./train_on_runpod.sh <runpod-ip> <runpod-port>

IP=$1
PORT=$2
KEY=~/.ssh/id_ed25519

SSH="ssh -o StrictHostKeyChecking=no root@$IP -p $PORT -i $KEY"

echo "🚀 Setting up superGPT on RunPod..."

# 1. Clone and setup
$SSH "cd /workspace && \
    git clone https://github.com/viralcode/superGPT.git 2>/dev/null; \
    cd superGPT && git pull && \
    pip install -q transformers datasets tiktoken numpy"

# 2. Prepare data (if not already done)
$SSH "cd /workspace/superGPT && \
    if [ ! -f data/train.bin ]; then \
        python -m supergpt.training.data_pipeline \
            --dataset HuggingFaceFW/fineweb-edu \
            --tokenizer Qwen/Qwen2.5-0.5B \
            --max-tokens 100000000 \
            --output-dir data/; \
    fi"

# 3. Start training in background
$SSH "cd /workspace/superGPT && \
    nohup python -u -m supergpt.training.train \
        --preset small \
        --data-dir data/ \
        --max-iters 10000 \
        --batch-size 32 \
        --lr 3e-4 \
        --compile \
        --device cuda \
        > /workspace/training.log 2>&1 &"

echo "✅ Training started!"
echo "📊 Monitor: $SSH 'tail -f /workspace/training.log'"
echo "🖥️  GPU:     $SSH 'nvidia-smi'"
```

---

## 13. Cost Optimization Tips

### 1. Use Spot Instances (40-70% Off)

Spot instances can be preempted but are much cheaper:

```
On-Demand A100: ~$1.64/hr
Spot A100:      ~$0.99/hr  (40% savings!)
```

> **Use for:** Quick experiments, data preparation, anything you can restart.
> **Don't use for:** Long training runs without checkpointing (use nohup + checkpoint resuming).

### 2. Use the Smallest GPU That Works

| Task | Cheapest GPU | Cost |
|------|-------------|------|
| Data prep | Any (CPU works) | $0.20/hr |
| Train ≤124M params | RTX 4090 | $0.44/hr |
| Train ≤350M params | A100 40GB | $1.10/hr |
| Train ≤1B params | A100 80GB | $1.64/hr |

### 3. Use `torch.compile()` (Free 2× Speedup)

Always add `--compile`. It halves your bill by doubling throughput.

### 4. Stop Your Pod When Not Training

RunPod charges per second. **Stop your pod** when you're not using it. Your data persists on the volume disk.

### 5. Estimate Cost Before Training

```python
# Quick cost estimator
tokens_per_second = 60000   # With --compile on A100
cost_per_hour = 1.64        # A100 80GB

total_tokens = 100_000_000  # 100M tokens
max_iters = 10000
batch_size = 32
block_size = 256

tokens_per_iter = batch_size * block_size  # 8,192
total_time_seconds = max_iters * tokens_per_iter / tokens_per_second
total_hours = total_time_seconds / 3600
total_cost = total_hours * cost_per_hour

print(f"Estimated time: {total_hours:.1f} hours")
print(f"Estimated cost: ${total_cost:.2f}")
# → Estimated time: 0.4 hours
# → Estimated cost: $0.60
```

---

## 14. Troubleshooting

### Pod Won't Start

- **"No available machines"** → Try a different GPU type or region
- **"Insufficient funds"** → Add more credits in Billing

### SSH Connection Refused

```bash
# Make sure pod is running (green status in dashboard)
# Try the web terminal in RunPod dashboard first
# Check that you're using the right port (not 22)
ssh root@<IP> -p <PORT> -i ~/.ssh/id_ed25519
```

### Training Killed When SSH Disconnects

```bash
# ALWAYS use nohup or tmux:
nohup python -u -m supergpt.training.train ... > /workspace/log.log 2>&1 &

# OR use tmux:
tmux new -s train
python -m supergpt.training.train ...
# Press Ctrl+B, then D to detach
# Reconnect later: tmux attach -t train
```

### CUDA Out of Memory

```bash
# Reduce batch size:
--batch-size 16  # instead of 32 or 64

# Use gradient accumulation:
--batch-size 8 --grad-accum 4  # effective batch = 32

# Use a smaller preset:
--preset small  # instead of medium or large

# Enable gradient checkpointing:
--gradient-checkpointing
```

### Slow Training (< 20K tok/s)

```bash
# Add --compile (most impactful)
--compile

# Increase batch size to saturate the GPU
--batch-size 64

# Check GPU utilization
nvidia-smi  # Should be > 80%
```

### Data Preparation Hangs

```bash
# HuggingFace datasets download can be slow
# Try with a subset first:
--max-tokens 10000000  # 10M tokens (quick test)

# Or use the sample subset:
--subset sample-10BT
```

### Checkpoint Too Large to Download

```bash
# Compress before downloading:
# On RunPod:
tar czf /workspace/model.tar.gz -C /workspace/superGPT/checkpoints best.pt

# On local machine:
scp -P <PORT> root@<IP>:/workspace/model.tar.gz ./
tar xzf model.tar.gz
```

---

## Quick Reference Card

```bash
# ──────────────────────────────────────────────────
# superGPT RunPod Cheat Sheet
# ──────────────────────────────────────────────────

# SSH into RunPod
ssh root@<IP> -p <PORT> -i ~/.ssh/id_ed25519

# Setup (first time only)
cd /workspace && git clone https://github.com/viralcode/superGPT.git
cd superGPT && pip install transformers datasets tiktoken numpy

# Prepare data
python -m supergpt.training.data_pipeline \
    --dataset HuggingFaceFW/fineweb-edu \
    --tokenizer Qwen/Qwen2.5-0.5B \
    --max-tokens 100000000 --output-dir data/

# Train (background, survives SSH disconnect)
nohup python -u -m supergpt.training.train \
    --preset small --data-dir data/ --max-iters 10000 \
    --batch-size 32 --lr 3e-4 --compile --device cuda \
    > /workspace/training.log 2>&1 &

# Monitor
tail -f /workspace/training.log
nvidia-smi

# Generate text
python -m supergpt.inference.generate \
    --checkpoint checkpoints/best.pt \
    --prompt "Hello world" --max-tokens 200

# Download model (from local machine)
scp -P <PORT> root@<IP>:/workspace/superGPT/checkpoints/best.pt ./
```
