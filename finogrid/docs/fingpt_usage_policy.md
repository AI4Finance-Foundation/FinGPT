# FinGPT Usage Policy for Finogrid v1

**Decision: What we keep, what we cut, and why.**

---

## The Honest Assessment

Finogrid v1 is a **payout orchestration platform**, not an AI research project.
FinGPT is a 50+ module LLM research framework built for academic and enterprise ML teams.

Most of FinGPT is irrelevant to what we're building. We use it as a source of
**pre-trained model weights and inference patterns only** — we do not retrain, do not
run distributed training clusters, and do not adopt its full framework.

---

## What We KEEP from FinGPT

### 1. Pre-trained Model Weights (HuggingFace Hub — no code, just model files)

| Model | Use in Finogrid | Why |
|-------|----------------|-----|
| `FinGPT/fingpt-sentiment_llama2-13b_lora` | Ops Oversight Agent: score corridor news | Best financial sentiment model available; no retraining needed |
| `FinGPT/fingpt-forecaster_dow30_llama2-7b_lora` | Process Improvement Agent: corridor signals | Adapted for FX/macro signal generation |

These are **loaded at agent startup** using `transformers` + `peft`. We do not copy
FinGPT training code — we just call `PeftModel.from_pretrained()`.

### 2. Inference Pattern (copied and simplified)

From `FinGPT_Sentiment_Analysis_v3/` we borrow:
- The prompt template: `"What is the sentiment of this news? {{positive/negative/neutral}}"`
- The generation config: greedy decode, max 10 tokens

That's 10 lines of code. Nothing else from that module.

### 3. RAG Pattern (from `FinGPT_RAG/`)

We borrow the **concept** of multi-source retrieval + embedding search.
Our implementation (`fingpt_integration/rag/knowledge_base.py`) is a clean rewrite
using ChromaDB directly — simpler than FinGPT's full RAG stack.

### 4. Data fetching utilities (concepts only)

From `FinGPT_Forecaster/data.py` we borrow:
- `yfinance` for FX and stablecoin price data
- `finnhub-python` for crypto/macro news headlines

We do **not** copy the GPT-4 labeling pipeline or the DOW30-specific data logic.

---

## What We CUT from FinGPT (and why)

| Module | What it does | Why we cut it |
|--------|-------------|---------------|
| `FinGPT_Benchmark/` | Multi-task instruction tuning + benchmarking framework | We don't retrain models. Irrelevant. |
| `FinGPT_FinancialReportAnalysis/` | 10-K SEC report PDF analysis | We don't analyze SEC filings. Irrelevant. |
| `FinGPT_MultiAgentsRAG/` | Multi-agent debate framework for hallucination reduction | Over-engineered. We have 5 simple agents. |
| `FinGPT_Sentiment_Analysis_v1/` | Older sentiment models (ChatGLM2, early Llama2) | Superseded by v3. |
| `FinGPT_Others/` | Trading strategies, Robo-Advisor, low-code dev | We're a payout platform, not a trading platform. |
| Training notebooks (4 root-level .ipynb) | Training walkthroughs | We don't train models. |
| DeepSpeed configs | Distributed training | 8×A100 GPU training is not in our budget or roadmap. |
| `setup.py`, `MANIFEST.in` | Python package distribution for FinGPT | We don't publish FinGPT as a package. |
| `requirements.txt` (root FinGPT) | FinGPT root deps (tushare, etc.) | Replaced by `finogrid/requirements.txt`. |

---

## Our Actual Dependency on FinGPT Code

```
FinGPT code we actually import or copy: ~30 lines
FinGPT model weights we use: 2 pre-trained LoRA adapters (HuggingFace)
FinGPT we delete or ignore: ~95% of the repo
```

---

## Practical Model Deployment for MVP

**Don't run FinGPT models in the hot path.** Here's the practical setup:

```
MVP Day 1 (no GPU budget):
  → Use OpenAI GPT-3.5-turbo with the FinGPT sentiment prompt template
  → Same prompt, 1/10th the infrastructure cost
  → Deploy when you have 100+ clients

MVP Day 2 (small GPU budget ~$50/month):
  → Run fingpt-sentiment_llama2-13b_lora on a single A10G via Modal or RunPod
  → Call it from agents only (not hot path)
  → 4-bit quantization (QLoRA) fits on 12GB VRAM

Production (funded, 1000+ clients):
  → Cloud Run GPU or Vertex AI endpoint
  → Load fingpt-forecaster alongside sentiment
  → Add FinGPT RAG with Vertex AI Matching Engine
```

---

## Summary

Finogrid v1 needs FinGPT for **3 things**:

1. Financial news sentiment scoring (pre-trained model, inference only)
2. FX/macro signal generation (adapted forecaster, inference only)
3. RAG pattern for internal support agent (concept, rewritten simply)

Everything else in the FinGPT repo can be ignored for v1.
