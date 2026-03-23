"""
FinGPT Sentiment Adapter for Finogrid.

Uses FinGPT's fine-tuned Llama-2 sentiment model to score crypto and
corridor-specific financial news. Results feed the Ops & Process Improvement agents.

NOT used in the hot transaction path — runs asynchronously.
"""
from __future__ import annotations

import os
import structlog
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import torch

# Increase HuggingFace Hub timeout to handle slow network connections or large file downloads
os.environ.setdefault("HF_HUB_TIMEOUT", "120")

log = structlog.get_logger()

# FinGPT sentiment labels
SENTIMENT_MAP = {
    "positive": 1,
    "negative": -1,
    "neutral": 0,
}

TEMPLATE = (
    "Instruction: What is the sentiment of this news? "
    "Please choose an answer from {{positive/negative/neutral}}.\n"
    "Input: {text}\n"
    "Answer: "
)


class FinoGridSentimentAnalyzer:
    """
    Thin wrapper around FinGPT's sentiment pipeline.
    Scores financial news relevant to Finogrid's corridors and crypto assets.
    """

    def __init__(
        self,
        base_model: str = "NousResearch/Llama-2-13b-hf",
        lora_model: str = "FinGPT/fingpt-sentiment_llama2-13b_lora",
        device: str = "auto",
    ):
        self.base_model = base_model
        self.lora_model = lora_model
        self.device = device
        self._model = None
        self._tokenizer = None

    def load(self):
        """Load the FinGPT sentiment model (lazy — call once at agent startup)."""
        log.info("fingpt_sentiment_loading", base=self.base_model, lora=self.lora_model)
        self._tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            trust_remote_code=True,
            device_map=self.device,
            torch_dtype=torch.float16,
        )
        self._model = PeftModel.from_pretrained(base, self.lora_model)
        self._model.eval()
        log.info("fingpt_sentiment_loaded")

    def score(self, text: str) -> dict:
        """
        Score a single piece of financial text.
        Returns: {"label": "positive"|"negative"|"neutral", "score": 1|0|-1}
        """
        if not self._model:
            raise RuntimeError("Model not loaded. Call load() first.")

        prompt = TEMPLATE.format(text=text[:512])
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        decoded = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = decoded[len(prompt):].strip().lower()

        # Parse label
        label = "neutral"
        for key in SENTIMENT_MAP:
            if key in answer:
                label = key
                break

        return {"label": label, "score": SENTIMENT_MAP[label], "raw": answer}

    def score_batch(self, texts: list[str]) -> list[dict]:
        """Score multiple texts. Returns list of results in same order."""
        return [self.score(t) for t in texts]

    def score_corridor_news(self, news_items: list[dict], corridor_code: str) -> list[dict]:
        """
        Score a list of news items relevant to a corridor.
        Each news_item should have: {"headline": str, "summary": str, "date": str}
        Returns enriched list with sentiment fields added.
        """
        results = []
        for item in news_items:
            text = f"{item.get('headline', '')}. {item.get('summary', '')}"
            sentiment = self.score(text)
            results.append({
                **item,
                "corridor": corridor_code,
                "sentiment_label": sentiment["label"],
                "sentiment_score": sentiment["score"],
            })
        return results
