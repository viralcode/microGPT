# superGPT Large — Trained Model

## Model Details

| Property | Value |
|----------|-------|
| **Architecture** | GPT-4 style Transformer (GQA 16Q/4KV, RoPE, SwiGLU) |
| **Parameters** | 421.1M (24 layers, 16 heads, 1024 embed) |
| **Tokenizer** | Qwen/Qwen2.5-0.5B (vocab_size: 151,643) |
| **Dataset** | 1B tokens of FineWeb-Edu (educational web text) |
| **Training** | 30,000 iters, cosine LR (1.5e-4), batch 8, block 2048 |
| **Best Val Loss** | 3.5834 |
| **Hardware** | NVIDIA A40 46GB, ~14 hours |
| **Date** | March 27, 2026 |

## Files

- `best.pt` — Model checkpoint (iter 29,500, best validation loss)
- `meta.pkl` — Tokenizer metadata (vocab size, tokenizer type)

## Usage

```bash
# From the superGPT project root:
python scripts/generate.py \
    --checkpoint trained_models/large_350m_fineweb/best.pt \
    --prompt "Your prompt here" \
    --max-tokens 200 \
    --temperature 0.7
```

## Capabilities

- ✅ Fluent English text generation
- ✅ Coherent multi-sentence paragraphs
- ✅ Educational/article writing style
- ⚠️ Hallucinated facts (expected at this scale)
- ⚠️ Occasional repetition in longer outputs

## License

Trained using the [superGPT](https://github.com/viralcode/superGPT) framework.
