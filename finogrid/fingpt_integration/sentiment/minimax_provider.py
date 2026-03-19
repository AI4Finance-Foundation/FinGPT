"""
MiniMax provider for FinGPT sentiment — cost-effective alternative to OpenAI.

Uses MiniMax's OpenAI-compatible API with MiniMax-M2.7 (enhanced reasoning and coding).
Same interface as OpenAISentimentFallback — drop-in replacement.

MiniMax API docs: https://platform.minimaxi.com/
"""
from __future__ import annotations

import os
import structlog
from openai import AsyncOpenAI

log = structlog.get_logger()

SENTIMENT_PROMPT = (
    "Instruction: What is the sentiment of this news? "
    "Please choose an answer from {{positive/negative/neutral}}.\n"
    "Input: {text}\n"
    "Answer:"
)

SENTIMENT_MAP = {"positive": 1, "negative": -1, "neutral": 0}

MINIMAX_BASE_URL = "https://api.minimax.io/v1"


class MiniMaxSentimentProvider:
    """
    Drop-in replacement for OpenAISentimentFallback using MiniMax API.
    Same interface — swap by setting FINGPT_LLM_PROVIDER=minimax.

    MiniMax's API is OpenAI-compatible, so we reuse the openai SDK
    with a custom base_url.
    """

    def __init__(self, model: str = "MiniMax-M2.7"):
        self.model = model
        api_key = os.getenv("MINIMAX_API_KEY")
        if not api_key:
            raise ValueError(
                "MINIMAX_API_KEY environment variable is required. "
                "Get your API key at https://platform.minimaxi.com/"
            )
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=MINIMAX_BASE_URL,
        )

    def load(self):
        pass  # Nothing to load for MiniMax

    async def score(self, text: str) -> dict:
        prompt = SENTIMENT_PROMPT.format(text=text[:512])
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
                # MiniMax requires temperature in (0.0, 1.0]; use 0.01 for near-deterministic output
                temperature=0.01,
            )
            answer = response.choices[0].message.content.strip().lower()
            label = "neutral"
            for key in SENTIMENT_MAP:
                if key in answer:
                    label = key
                    break
            return {"label": label, "score": SENTIMENT_MAP[label], "raw": answer}
        except Exception as e:
            log.error("minimax_sentiment_failed", error=str(e))
            return {"label": "neutral", "score": 0, "error": str(e)}

    async def score_corridor_news(self, news_items: list[dict], corridor_code: str) -> list[dict]:
        results = []
        for item in news_items:
            text = f"{item.get('headline', '')}. {item.get('summary', '')}"
            sentiment = await self.score(text)
            results.append({
                **item,
                "corridor": corridor_code,
                "sentiment_label": sentiment["label"],
                "sentiment_score": sentiment["score"],
            })
        return results
