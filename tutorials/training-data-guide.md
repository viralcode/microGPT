# How to Prepare Training Data for a Frontier LLM

**A complete guide to building the data pipeline behind models like GPT-4, Claude, DeepSeek-V3, and Llama 3 — from crawling raw internet data to producing a training-ready token stream.**

---

## Table of Contents

1. [Overview: What Goes Into an LLM](#1-overview-what-goes-into-an-llm)
2. [Phase 1: Web Crawling](#2-phase-1-web-crawling)
3. [Phase 2: Text Extraction](#3-phase-2-text-extraction)
4. [Phase 3: Quality Filtering](#4-phase-3-quality-filtering)
5. [Phase 4: Deduplication](#5-phase-4-deduplication)
6. [Phase 5: Domain-Specific Data](#6-phase-5-domain-specific-data)
7. [Phase 6: Tokenization](#7-phase-6-tokenization)
8. [Phase 7: Data Mixing & Packaging](#8-phase-7-data-mixing--packaging)
9. [Phase 8: Training at Scale](#9-phase-8-training-at-scale)
10. [Phase 9: Post-Training (SFT + RLHF)](#10-phase-9-post-training-sft--rlhf)
11. [Appendix: Open Datasets](#11-appendix-open-datasets)

---

## 1. Overview: What Goes Into an LLM

A frontier LLM requires **trillions of tokens** of diverse, high-quality data. Here's what the leading labs actually use:

### Data Mixture Ratios (from public technical reports)

| Model | Total Tokens | General Text | Code | Math & Reasoning | Multilingual |
|-------|-------------|-------------|------|------------------|-------------|
| **Llama 3** | 15T | 50% | 17% | 25% | 8% |
| **DeepSeek-V3** | 14.8T | ~55% | ~20% | ~15% | ~10% |
| **GPT-4** | ~13T (est.) | ~50% | ~20% | ~20% | ~10% |

### The Pipeline at a Glance

```
┌──────────────┐    ┌───────────────┐    ┌──────────────┐
│  RAW CRAWL   │───▶│  EXTRACTION   │───▶│  FILTERING   │
│  (petabytes) │    │  HTML → Text  │    │  Quality/PII │
└──────────────┘    └───────────────┘    └──────────────┘
                                               │
                    ┌───────────────┐    ┌──────▼───────┐
                    │   TOKENIZE    │◀───│   DEDUP      │
                    │  BPE/SentPc   │    │  MinHash/LSH │
                    └───────┬───────┘    └──────────────┘
                            │
                    ┌───────▼───────┐    ┌──────────────┐
                    │  MIX & PACK   │───▶│   TRAINING   │
                    │  Ratios/Shards│    │  FSDP/DeepSp │
                    └───────────────┘    └──────────────┘
```

---

## 2. Phase 1: Web Crawling

### Option A: Use Common Crawl (Recommended)

[Common Crawl](https://commoncrawl.org/) is a free, open repository of web crawl data containing **petabytes** of web pages. This is what FineWeb, RedPajama, The Pile, and most open LLM datasets are built from.

```bash
# Common Crawl data is on AWS S3 (free, no auth required)
# Each monthly crawl = ~3 billion pages

# List available crawls
aws s3 ls s3://commoncrawl/ --no-sign-request

# Download a WARC segment
aws s3 cp s3://commoncrawl/crawl-data/CC-MAIN-2024-10/segments/... \
    ./data/raw/ --no-sign-request
```

**File formats:**
- **WARC** (Web ARChive): Raw HTTP responses + HTML — **use this one**
- **WET**: Pre-extracted plain text (lower quality)
- **WAT**: Metadata only (URLs, timestamps)

> [!TIP]
> Always process WARC files yourself with Trafilatura rather than using WET files. Your text quality will be dramatically better.

### Option B: Custom Crawling

```python
"""Custom web crawler for specialized domains."""
import scrapy

class LLMDataSpider(scrapy.Spider):
    name = "llm_data"
    custom_settings = {
        'ROBOTSTXT_OBEY': True,       # Always respect robots.txt
        'DOWNLOAD_DELAY': 1.0,        # 1 req/sec (be polite)
        'CONCURRENT_REQUESTS': 8,
        'DEPTH_LIMIT': 3,
    }
    
    def parse(self, response):
        text = ' '.join(response.css('p::text, h1::text, h2::text').getall())
        if len(text) > 200:
            yield {'url': response.url, 'text': text}
        for href in response.css('a::attr(href)').getall():
            yield response.follow(href, self.parse)
```

**Crawling tools comparison:**

| Tool | Best For | Scale |
|------|---------|-------|
| [Common Crawl](https://commoncrawl.org/) | Full web corpus, free | Petabyte |
| [Scrapy](https://scrapy.org/) | Custom domain crawls | Medium |
| [Trafilatura](https://trafilatura.readthedocs.io/) | Text extraction from HTML | Any |
| [Firecrawl](https://firecrawl.dev/) | JS-rendered pages | Small-Med |

> [!CAUTION]
> **Always check:** robots.txt, Terms of Service, copyright/licensing, GDPR/CCPA for personal data. Use permissively-licensed data when possible.

---

## 3. Phase 2: Text Extraction

Raw HTML is full of menus, ads, JavaScript. You need only the **main content**.

```python
"""Extract clean text from HTML using Trafilatura (industry standard)."""
import trafilatura

def extract_from_warc(warc_path):
    """Process a Common Crawl WARC file."""
    import warcio
    with open(warc_path, 'rb') as f:
        for record in warcio.ArchiveIterator(f):
            if record.rec_type == 'response':
                html = record.content_stream().read().decode('utf-8', errors='ignore')
                text = trafilatura.extract(html, favor_precision=True)
                if text and len(text) > 200:
                    yield {
                        'url': record.rec_headers.get('WARC-Target-URI'),
                        'text': text,
                    }
```

For **petabyte scale**, use Apache Spark on AWS EMR:

```python
# Spark job: distribute across 1000+ executors
warc_paths = spark.read.text("s3://commoncrawl/.../warc.paths.gz")
extracted = warc_paths.rdd.flatMap(extract_from_warc)
extracted.saveAsTextFile("s3://my-bucket/extracted/")
```

---

## 4. Phase 3: Quality Filtering

This is the **most impactful step**. Quality > quantity.

### Layer 1: Heuristic Filters (Fast, Rule-Based)

```python
"""Based on FineWeb, RedPajama, and DeepMind Gopher heuristics."""
import re
from collections import Counter

def heuristic_filter(text: str) -> tuple[bool, str]:
    """Returns (pass, reason). Document must pass ALL checks."""
    words = text.split()
    n_words = len(words)
    
    if n_words < 50:                    return False, "too_short"
    if n_words > 100_000:               return False, "too_long"
    
    # Average word length (catches garbled text)
    avg_wl = sum(len(w) for w in words) / n_words
    if avg_wl < 3 or avg_wl > 10:       return False, "unusual_words"
    
    # Alpha ratio (catches code dumps, symbol spam)
    alpha = sum(c.isalpha() for c in text)
    if alpha / max(len(text), 1) < 0.5: return False, "low_alpha"
    
    # Line repetition
    lines = text.split('\n')
    if len(set(lines)) / max(len(lines), 1) < 0.5:
        return False, "too_repetitive"
    
    # Top bigram frequency (from Gopher paper)
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
    bc = Counter(bigrams)
    if bc and bc.most_common(1)[0][1] / len(bigrams) > 0.2:
        return False, "repeated_ngrams"
    
    # URL density (catches link farms)
    urls = re.findall(r'https?://\S+', text)
    if len(urls) / max(n_words, 1) > 0.1:
        return False, "too_many_urls"
    
    return True, "passed"
```

### Layer 2: Language Detection

```python
"""Filter by language using fastText (176 languages)."""
import fasttext
model = fasttext.load_model('lid.176.bin')

def detect_language(text):
    pred = model.predict(text.replace('\n', ' ')[:1000], k=1)
    return pred[0][0].replace('__label__', ''), pred[1][0]

def is_english(text, min_conf=0.65):
    lang, conf = detect_language(text)
    return lang == 'en' and conf >= min_conf
```

### Layer 3: Model-Based Quality Scoring (The Secret Weapon)

This is what separates FineWeb-Edu from FineWeb — and why it trains dramatically better models:

```python
"""
The FineWeb-Edu approach:
1. Use Llama-3-70B to score 500K samples on educational quality (0-5)
2. Train a fast fastText classifier on those labels
3. Run classifier on the full 15T token corpus
4. Keep only documents scoring ≥ 3

This single step improved downstream benchmarks by 10-20%.
"""
QUALITY_PROMPT = """Rate the educational quality of this text (0-5):
0 = Spam/ads    1 = Social media    2 = Average web content
3 = Informative  4 = Educational    5 = Textbook-quality

Text: {text}
Score:"""

# Score 500K samples with a big LLM → train fastText → filter everything
# pip install fasttext
# fasttext supervised -input labels.txt -output quality_model -epoch 25
```

### Layer 4: PII Removal

```python
"""Remove personal data (email, phone, SSN, credit cards)."""
import re

PII_PATTERNS = {
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
    'ssn':   r'\b\d{3}-\d{2}-\d{4}\b',
    'cc':    r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
}

def redact_pii(text):
    for name, pattern in PII_PATTERNS.items():
        text = re.sub(pattern, f'[{name.upper()}_REDACTED]', text)
    return text
```

---

## 5. Phase 4: Deduplication

Web crawls contain **30-50% duplicates**. Training on duplicates wastes compute and causes memorization.

### Exact Dedup (Line/Document Level)

```python
import hashlib

def exact_dedup(documents):
    seen = set()
    for doc in documents:
        h = hashlib.sha256(doc['text'].encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            yield doc
```

### Fuzzy Dedup: MinHash + LSH (Industry Standard)

Used by FineWeb, RedPajama, The Pile, Dolma. Finds documents that are ~80%+ similar.

```python
"""
MinHash LSH — the industry standard for near-duplicate detection.
1. Convert documents to n-gram sets (shingles)
2. Compute MinHash signatures (compact fingerprints)
3. Use LSH banding to find candidate pairs
4. Remove near-duplicates (Jaccard similarity > 0.8)
"""
from datasketch import MinHash, MinHashLSH

def create_minhash(text, num_perm=128):
    m = MinHash(num_perm=num_perm)
    words = text.lower().split()
    for i in range(len(words) - 4):
        shingle = ' '.join(words[i:i+5])
        m.update(shingle.encode('utf-8'))
    return m

def fuzzy_dedup(documents, threshold=0.8):
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    minhashes = {}
    
    for doc in documents:
        mh = create_minhash(doc['text'])
        minhashes[doc['id']] = mh
        try:
            lsh.insert(doc['id'], mh)
        except ValueError:
            pass  # Already exists
    
    # Find duplicates
    duplicates = set()
    for doc_id, mh in minhashes.items():
        if doc_id in duplicates: continue
        for candidate in lsh.query(mh):
            if candidate != doc_id:
                duplicates.add(candidate)
    
    return [d for d in documents if d['id'] not in duplicates]
```

For **trillion-token scale**, use suffix array dedup (finds duplicate **substrings** across documents):

```bash
# pip install text-dedup
python -m text_dedup.suffix_array --input corpus.jsonl --output deduped.jsonl --k 50
```

---

## 6. Phase 5: Domain-Specific Data

### 6.1 Code (17-20% of training mix)

Code dramatically improves reasoning — even for non-code tasks.

**Primary source: [The Stack v2](https://huggingface.co/datasets/bigcode/the-stack-v2)** — 6.4 TB, 358 languages, permissively licensed.

```python
from datasets import load_dataset

ds = load_dataset("bigcode/the-stack-v2", split="train", streaming=True)

def filter_code(example):
    code = example['content']
    if len(code) < 100 or len(code) > 100_000: return False
    # Skip auto-generated
    if any(m in code[:500] for m in ['auto-generated', 'DO NOT EDIT']): return False
    # Skip minified
    if len(code) / max(code.count('\n'), 1) > 200: return False
    return True

# Language weights (by impact on reasoning)
LANG_WEIGHTS = {
    'python': 0.25, 'javascript': 0.12, 'typescript': 0.10,
    'java': 0.10, 'c++': 0.08, 'c': 0.06, 'go': 0.05,
    'rust': 0.05, 'shell': 0.03, 'sql': 0.03,
}
```

### 6.2 Math (15-25% of training mix)

Mathematical data is critical for reasoning transfer.

| Dataset | Size | Content | Use |
|---------|------|---------|-----|
| [OpenWebMath](https://huggingface.co/datasets/open-web-math/open-web-math) | 14.7B tokens | Math from web (LaTeX, proofs) | Pre-training |
| [MathPile](https://huggingface.co/datasets/GAIR/MathPile) | 9.5B tokens | Textbooks, arXiv, StackExchange | Pre-training |
| [GSM8K](https://huggingface.co/datasets/openai/gsm8k) | 8.5K problems | Grade school word problems | SFT |
| [MATH](https://huggingface.co/datasets/hendrycks/competition_math) | 12.5K problems | Competition math (AMC, Olympiad) | SFT |
| [NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) | 860K problems | CoT solutions, all levels | SFT |

```python
# Key: preserve LaTeX notation during processing!
def process_math(text):
    # DON'T strip $...$ or \[...\] — these teach math reasoning
    text = re.sub(r'\n{3,}', '\n\n', text)
    has_math = any(m in text for m in ['$', '\\frac', '\\sum', 'theorem', 'proof'])
    return text if has_math else None
```

### 6.3 Academic Papers

- **arXiv**: 2M+ papers with full LaTeX source — [Kaggle mirror](https://www.kaggle.com/datasets/Cornell-University/arxiv)
- **S2ORC**: 81M papers across all fields — [GitHub](https://github.com/allenai/s2orc)
- **PubMed**: 35M+ biomedical abstracts

### 6.4 Books & Long-Form

- **Project Gutenberg**: 70K+ public domain books — [gutenberg.org](https://www.gutenberg.org/)
- Critical for teaching long-range coherence

### 6.5 Conversation & Instruction Data (Post-Training)

| Dataset | Size | Quality |
|---------|------|---------|
| [OpenAssistant/oasst2](https://huggingface.co/datasets/OpenAssistant/oasst2) | ~50M tok | High |
| [WizardLM Evol-Instruct](https://huggingface.co/datasets/WizardLMTeam/WizardLM_evol_instruct_V2_196k) | ~100M tok | High |
| [UltraChat](https://huggingface.co/datasets/stingning/ultrachat) | ~500M tok | Medium |
| [Orca-Math](https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k) | ~50M tok | High |

---

## 7. Phase 6: Tokenization

### Choosing a Tokenizer

| Tokenizer | Vocab Size | Used By | Notes |
|-----------|-----------|---------|-------|
| **tiktoken** (cl100k) | 100K | GPT-4, GPT-3.5 | Fast, byte-level BPE |
| **SentencePiece** | 32K-128K | LLaMA, Mistral | Good multilingual support |
| **Qwen tokenizer** | 151K | Qwen2.5, DeepSeek-V3 | Best for code + multilingual |

### Training a Custom Tokenizer

```python
"""Train a BPE tokenizer optimized for YOUR data mix."""
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

trainer = trainers.BpeTrainer(
    vocab_size=100_000,
    min_frequency=100,
    special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
    show_progress=True,
)

# Train on a representative sample of your data mix
tokenizer.train(files=[
    "data/web_sample.txt",     # 50%
    "data/code_sample.txt",    # 20%
    "data/math_sample.txt",    # 20%
    "data/multilingual.txt",   # 10%
], trainer=trainer)

tokenizer.save("my_tokenizer.json")
```

### Convert Text → Token IDs → Binary

```python
"""Convert filtered text corpus into binary training files."""
import numpy as np
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
vocab_size = tokenizer.vocab_size  # 151,643

# Choose dtype based on vocab size
dtype = np.uint32 if vocab_size > 65535 else np.uint16

all_tokens = []
for doc in read_jsonl("filtered_corpus.jsonl"):
    tokens = tokenizer.encode(doc['text'])
    tokens.append(tokenizer.eos_token_id)  # Document separator
    all_tokens.extend(tokens)

# Save as memory-mapped binary
arr = np.array(all_tokens, dtype=dtype)
arr.tofile("data/train.bin")

print(f"Saved {len(arr):,} tokens ({arr.nbytes / 1e9:.1f} GB)")
```

> [!IMPORTANT]
> **Always use `uint32` if your vocab size > 65,535!** Using `uint16` for large vocabs (like Qwen's 151K) will silently corrupt your data by truncating token IDs. This is a common bug — we found and fixed it in superGPT.

---

## 8. Phase 7: Data Mixing & Packaging

### The Art of Data Mixing

The ratio of different data domains **dramatically** affects model capabilities. Here's the recipe:

```python
"""
Data mixing strategy based on Llama 3 and DeepSeek-V3.

Key insights:
1. Start with ~50% general web text (foundation)
2. Code improves reasoning even on non-code tasks (17-20%)
3. Math is the highest-leverage domain for reasoning (15-25%)
4. Overtrain on high-quality data in later phases (annealing)
"""

# Phase 1: Main pre-training (first 80% of compute)
PRETRAIN_MIX = {
    'web_text':     0.50,   # FineWeb-Edu, RedPajama, C4
    'code':         0.17,   # The Stack, GitHub
    'math':         0.15,   # OpenWebMath, MathPile
    'academic':     0.08,   # arXiv, S2ORC, PubMed
    'books':        0.05,   # Gutenberg, BookCorpus
    'multilingual': 0.05,   # CC-100, OPUS
}

# Phase 2: Annealing (final 20% of compute)
# Upsample high-quality data for final polish
ANNEAL_MIX = {
    'web_text':     0.30,   # Only top-quality (FineWeb-Edu score ≥ 4)
    'code':         0.25,   # Upsample — code helps reasoning
    'math':         0.25,   # Heavy upsample on math
    'academic':     0.10,
    'books':        0.05,
    'multilingual': 0.05,
}
```

### Sharding for Distributed Training

```python
"""
Shard data for multi-GPU/multi-node training.
Each shard should be ~256 MB - 1 GB.
"""
import numpy as np
import os

def shard_data(input_path, output_dir, n_shards=256):
    """Split a large .bin file into evenly-sized shards."""
    os.makedirs(output_dir, exist_ok=True)
    
    data = np.memmap(input_path, dtype=np.uint32, mode='r')
    shard_size = len(data) // n_shards
    
    for i in range(n_shards):
        start = i * shard_size
        end = start + shard_size if i < n_shards - 1 else len(data)
        
        shard = np.array(data[start:end], dtype=np.uint32)
        shard.tofile(os.path.join(output_dir, f"shard_{i:05d}.bin"))
    
    print(f"Created {n_shards} shards in {output_dir}")
    print(f"Shard size: ~{shard_size * 4 / 1e6:.0f} MB each")

# Example: 100B tokens → 400 GB → 400 shards of 1 GB each
shard_data("data/train.bin", "data/shards/", n_shards=400)
```

### Data Loading for Training

```python
"""
Efficient data loading for billion-token training runs.
Uses memory-mapped files — never loads the full dataset into RAM.
"""
import numpy as np
import torch

class ShardedDataLoader:
    """Load data from multiple shards, cycling through them."""
    
    def __init__(self, shard_dir, block_size, batch_size, dtype=np.uint32):
        self.shards = sorted(
            [os.path.join(shard_dir, f) for f in os.listdir(shard_dir)
             if f.endswith('.bin')]
        )
        self.block_size = block_size
        self.batch_size = batch_size
        self.dtype = dtype
        self.current_shard = 0
        self._load_shard()
    
    def _load_shard(self):
        self.data = np.memmap(
            self.shards[self.current_shard],
            dtype=self.dtype, mode='r'
        )
    
    def get_batch(self):
        ix = torch.randint(len(self.data) - self.block_size - 1, (self.batch_size,))
        x = torch.stack([
            torch.from_numpy(self.data[i:i+self.block_size].astype(np.int64))
            for i in ix
        ])
        y = torch.stack([
            torch.from_numpy(self.data[i+1:i+1+self.block_size].astype(np.int64))
            for i in ix
        ])
        
        # Advance shard every N batches
        self.current_shard = (self.current_shard + 1) % len(self.shards)
        self._load_shard()
        
        return x, y
```

---

## 9. Phase 8: Training at Scale

### Hardware Requirements

| Model Size | GPUs Needed | VRAM per GPU | Training Time | Estimated Cost |
|-----------|------------|-------------|--------------|---------------|
| 125M | 1× A100/H100 | 40 GB | 1-3 days | $100-500 |
| 1B | 4× A100 | 40 GB each | 1-2 weeks | $2K-5K |
| 7B | 8× A100 | 80 GB each | 2-4 weeks | $20K-50K |
| 70B | 64× A100 | 80 GB each | 1-2 months | $300K-1M |
| 405B (Llama 3) | 16K H100 | 80 GB each | 54 days | $30M+ |
| 671B (DeepSeek-V3) | 2048 H800 | 80 GB each | 2 months | $5.5M |

### Parallelism Strategies

```python
"""
At scale, you need multiple forms of parallelism:

1. Data Parallelism (DP): Same model on each GPU, different data
   - Simplest. Works up to ~8 GPUs.
   - PyTorch DDP or FSDP

2. Tensor Parallelism (TP): Split layers across GPUs
   - Splits attention heads and FFN across GPUs within a node
   - Requires fast interconnect (NVLink)

3. Pipeline Parallelism (PP): Split layers across nodes
   - Layer 0-15 on Node 1, Layer 16-31 on Node 2, etc.
   - Works across slower interconnects

4. Expert Parallelism (EP): For MoE models
   - Each GPU holds a subset of experts
   - Tokens are all-to-all dispatched to the right GPU
"""

# For most users: FSDP is the easiest path to multi-GPU
# torchrun --nproc_per_node=8 -m supergpt.training.train \
#     --preset medium --distributed --compile

# superGPT handles FSDP setup automatically:
from supergpt.training.train import main
# Just add --distributed flag
```

### Training with superGPT

```bash
# Single GPU (up to ~1B params on 80GB VRAM)
python -m supergpt.training.train \
    --preset small \
    --data-dir data \
    --max-iters 50000 \
    --batch-size 32 \
    --lr 3e-4 \
    --compile \
    --device cuda

# Multi-GPU with FSDP
torchrun --nproc_per_node=8 \
    -m supergpt.training.train \
    --preset medium \
    --data-dir data \
    --max-iters 100000 \
    --distributed \
    --compile

# With FP8 (2× memory savings on H100)
python -m supergpt.training.train \
    --preset large \
    --data-dir data \
    --fp8 \
    --compile

# With logging
python -m supergpt.training.train \
    --preset small \
    --wandb \
    --tensorboard
```

### Key Training Hyperparameters

| Parameter | Small (≤1B) | Medium (1-10B) | Large (≥10B) |
|-----------|------------|---------------|-------------|
| Learning Rate | 3e-4 | 1.5e-4 | 6e-5 |
| Batch Size (tokens) | 512K | 2M | 4M+ |
| Warmup Steps | 2,000 | 2,000 | 2,000 |
| LR Schedule | Cosine | Cosine | Cosine |
| Weight Decay | 0.1 | 0.1 | 0.1 |
| Gradient Clipping | 1.0 | 1.0 | 1.0 |
| Context Length | 2K → 8K | 4K → 32K | 4K → 128K |

---

## 10. Phase 9: Post-Training (SFT + RLHF)

After pre-training, the model can predict text but can't follow instructions. Post-training turns it into an assistant.

### Stage 1: Supervised Fine-Tuning (SFT)

Train on 500K-1.5M instruction-following examples. The model learns the system/user/assistant turn pattern.

```bash
# superGPT SFT with LoRA (memory-efficient)
python -m supergpt.training.finetune \
    --checkpoint checkpoints/best.pt \
    --data-dir data/sft/ \
    --lora-rank 64 \
    --max-iters 5000 \
    --lr 2e-5
```

### Stage 2: RLHF / DPO Alignment

Align the model to human preferences. DPO (Direct Preference Optimization) is simpler than classic RLHF — no reward model needed.

```bash
# superGPT DPO alignment
python -m supergpt.training.align_dpo \
    --checkpoint checkpoints/sft_best.pt \
    --data-dir data/preferences/ \
    --beta 0.1 \
    --max-iters 2000
```

### Stage 3: Knowledge Distillation (Optional)

Transfer reasoning from a large teacher to a smaller student. DeepSeek-V3 distilled from DeepSeek-R1 for enhanced reasoning.

```bash
# superGPT distillation
python -m supergpt.training.distill \
    --teacher checkpoints/large_model.pt \
    --student-preset small \
    --data-dir data/ \
    --alpha 0.7 \
    --temperature 2.0
```

---

## 11. Appendix: Open Datasets

### Pre-Training Datasets

| Dataset | Size | Description | Link |
|---------|------|-------------|------|
| **FineWeb-Edu** | 1.3T tokens | Ultra-high-quality filtered web text | [HuggingFace](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) |
| **FineWeb** | 15T tokens | Full Common Crawl processed with quality filters | [HuggingFace](https://huggingface.co/datasets/HuggingFaceFW/fineweb) |
| **RedPajama-V2** | 30T tokens | Full web corpus with 40+ quality annotations | [Together AI](https://together.ai/blog/redpajama-data-v2) |
| **The Pile** | 825 GB | Curated mix of 22 diverse sources | [HuggingFace](https://huggingface.co/datasets/EleutherAI/pile) |
| **Dolma** | 3T tokens | Open corpus for OLMo (Allen AI) | [HuggingFace](https://huggingface.co/datasets/allenai/dolma) |
| **C4** | 750 GB | Cleaned Common Crawl (original T5 data) | [HuggingFace](https://huggingface.co/datasets/allenai/c4) |

### Code Datasets

| Dataset | Size | Description | Link |
|---------|------|-------------|------|
| **The Stack v2** | 6.4 TB | 358 programming languages, permissive licenses | [HuggingFace](https://huggingface.co/datasets/bigcode/the-stack-v2) |
| **StarCoder Data** | 783 GB | Curated code for StarCoder models | [HuggingFace](https://huggingface.co/datasets/bigcode/starcoderdata) |

### Math Datasets

| Dataset | Size | Description | Link |
|---------|------|-------------|------|
| **OpenWebMath** | 14.7B tokens | Math content filtered from Common Crawl | [HuggingFace](https://huggingface.co/datasets/open-web-math/open-web-math) |
| **MathPile** | 9.5B tokens | Textbooks, arXiv, StackExchange | [HuggingFace](https://huggingface.co/datasets/GAIR/MathPile) |
| **GSM8K** | 8.5K | Grade school math with solutions | [HuggingFace](https://huggingface.co/datasets/openai/gsm8k) |
| **MATH** | 12.5K | Competition math with step-by-step solutions | [HuggingFace](https://huggingface.co/datasets/hendrycks/competition_math) |
| **NuminaMath-CoT** | 860K | Chain-of-thought math solutions | [HuggingFace](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) |

### Instruction / SFT Datasets

| Dataset | Size | Description | Link |
|---------|------|-------------|------|
| **OpenAssistant** | 161K msgs | Human-written multi-turn conversations | [HuggingFace](https://huggingface.co/datasets/OpenAssistant/oasst2) |
| **UltraChat** | 1.5M convos | Multi-turn synthetic conversations | [HuggingFace](https://huggingface.co/datasets/stingning/ultrachat) |
| **WizardLM** | 196K | Evolved instructions for complex tasks | [HuggingFace](https://huggingface.co/datasets/WizardLMTeam/WizardLM_evol_instruct_V2_196k) |
| **Orca-Math** | 200K | Math word problems for reasoning | [HuggingFace](https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k) |

### Tools & Libraries

| Tool | Purpose | Link |
|------|---------|------|
| **text-dedup** | Deduplication at scale | [GitHub](https://github.com/ChenghaoMou/text-dedup) |
| **datasketch** | MinHash / LSH implementation | [GitHub](https://github.com/ekzhu/datasketch) |
| **Trafilatura** | Web text extraction | [GitHub](https://github.com/adbar/trafilatura) |
| **fastText** | Language detection + quality classification | [GitHub](https://github.com/facebookresearch/fastText) |
| **tokenizers** | Fast BPE/WordPiece/Unigram training | [GitHub](https://github.com/huggingface/tokenizers) |
| **datatrove** | HuggingFace's data processing toolkit | [GitHub](https://github.com/huggingface/datatrove) |

---

## References

- [Llama 3 Technical Report](https://arxiv.org/abs/2407.21783) — Meta, 2024
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — DeepSeek AI, 2024
- [FineWeb: 15T Tokens of High-Quality Web Data](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1) — HuggingFace, 2024
- [The Stack: 3TB of Source Code](https://arxiv.org/abs/2211.15533) — BigCode Project, 2022
- [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264) — Muennighoff et al., 2023
- [OpenWebMath: Open Dataset of High-Quality Mathematical Web Text](https://arxiv.org/abs/2310.06786) — Paster et al., 2023
- [RedPajama-V2: An Open Dataset with 30T Tokens](https://together.ai/blog/redpajama-data-v2) — Together AI, 2023
