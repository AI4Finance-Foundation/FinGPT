"""
FinGPT integration for Finogrid v1.

What we actually use from FinGPT:
1. Pre-trained sentiment model (fingpt-sentiment_llama2-13b_lora) — inference only
2. Pre-trained forecaster model (fingpt-forecaster_dow30_llama2-7b_lora) — inference only
3. RAG pattern — rewritten using ChromaDB

What we do NOT use:
- Training pipelines (FinGPT_Benchmark, DeepSpeed configs)
- Multi-agent debate framework (FinGPT_MultiAgentsRAG)
- SEC report analysis (FinGPT_FinancialReportAnalysis)
- Trading strategies (FinGPT_Others)
- v1 sentiment models (superseded by v3)

For MVP: set FINGPT_USE_OPENAI_FALLBACK=true to use GPT-3.5 with FinGPT prompts
instead of loading the full 13B model locally. Same quality for early stage.

LLM provider selection (for sentiment analysis and agent LLM client):
  FINGPT_LLM_PROVIDER=openai   → OpenAI GPT-3.5-turbo (default)
  FINGPT_LLM_PROVIDER=minimax  → MiniMax MiniMax-M2.7 (enhanced reasoning and coding)
  FINGPT_LLM_PROVIDER=fingpt   → Local FinGPT model (requires GPU)
"""
import os

USE_OPENAI_FALLBACK = os.getenv("FINGPT_USE_OPENAI_FALLBACK", "true").lower() == "true"

# Provider selection: "openai" (default), "minimax", or "fingpt" (local model)
LLM_PROVIDER = os.getenv("FINGPT_LLM_PROVIDER", "openai").lower()
