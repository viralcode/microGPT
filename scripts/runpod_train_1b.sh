#!/bin/bash
# =============================================================================
#  superGPT — 1B Token Training Run
#
#  Prepares 1B tokens from FineWeb-Edu (streaming, OOM-proof)
#  and trains the model for 50K iterations.
#
#  Budget: ~$3-4 total (3-4 hours on A40 @ $0.79/hr)
#  Expected: val loss ~2.0-2.5, high-quality English text
#
#  Usage:
#    chmod +x scripts/runpod_train_1b.sh
#    nohup bash scripts/runpod_train_1b.sh > /workspace/train.log 2>&1 &
# =============================================================================

set -e

echo "============================================================"
echo "  superGPT — 1B Token Training Pipeline"
echo "  Dataset: FineWeb-Edu (streaming, OOM-proof)"
echo "============================================================"

# ── Setup ────────────────────────────────────────────────────────────────────
cd /workspace

if [ ! -d "superGPT" ]; then
    git clone https://github.com/viralcode/superGPT.git
    cd superGPT
else
    cd superGPT
    git pull origin main
fi

pip install -q torch numpy transformers datasets tokenizers
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Step 1: Prepare 1B Tokens ────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Step 1: Streaming 1B tokens from FineWeb-Edu"
echo "  Shard pattern: 10M tokens/shard, constant 40MB RAM"
echo "  ETA: ~38 minutes"
echo "============================================================"

rm -f data/*.bin data/meta.pkl 2>/dev/null

python -u -m supergpt.training.data_pipeline \
    --dataset HuggingFaceFW/fineweb-edu \
    --subset sample-10BT \
    --tokenizer Qwen/Qwen2.5-0.5B \
    --output-dir data_1b \
    --max-tokens 1000000000 \
    --shard-size 10000000

echo ""
echo "Data preparation complete!"
ls -lh data_1b/

# ── Step 2: Train (50K iters) ────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Step 2: Training (small preset, ~69M params, 50K iters)"
echo "  ETA: ~2-3 hours on A40"
echo "============================================================"

python -u -m supergpt.training.train \
    --preset small \
    --data-dir data_1b \
    --max-iters 50000 \
    --batch-size 32 \
    --lr 3e-4 \
    --device cuda \
    --lr-schedule cosine \
    --gradient-checkpointing

# ── Step 3: Generate Text ────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Step 3: Text Generation Test"
echo "============================================================"

python scripts/generate.py \
    --checkpoint checkpoints/best.pt \
    --prompt "Artificial intelligence is" \
    --max-tokens 200 \
    --temperature 0.7

echo ""
python scripts/generate.py \
    --checkpoint checkpoints/best.pt \
    --prompt "The history of science began when" \
    --max-tokens 200 \
    --temperature 0.7

echo ""
echo "============================================================"
echo "  ALL DONE! 1B token training complete."
echo "============================================================"
