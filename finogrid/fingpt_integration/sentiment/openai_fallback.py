"""
OpenAI fallback for FinGPT sentiment — practical for MVP stage.

Uses the same FinGPT prompt template with GPT-3.5-turbo.
~$0.002 per 1000 calls. No GPU needed. Switch to full FinGPT model when scaling.
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


class OpenAISentimentFallback:
    """
    Drop-in replacement for FinoGridSentimentAnalyzer using OpenAI API.
    Same interface — swap with the full FinGPT model when budget allows.
    """

    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.model = model
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def load(self):
        pass  # Nothing to load for OpenAI

    async def score(self, text: str) -> dict:
        prompt = SENTIMENT_PROMPT.format(text=text[:512])
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0,
            )
            answer = response.choices[0].message.content.strip().lower()
            label = "neutral"
            for key in SENTIMENT_MAP:
                if key in answer:
                    label = key
                    break
            return {"label": label, "score": SENTIMENT_MAP[label], "raw": answer}
        except Exception as e:
            log.error("openai_sentiment_failed", error=str(e))
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


def get_sentiment_analyzer():
    """
    Factory: returns OpenAI fallback for MVP, full FinGPT model for production.
    Controlled by FINGPT_USE_OPENAI_FALLBACK env var.
    """
    from .. import USE_OPENAI_FALLBACK
    if USE_OPENAI_FALLBACK:
        log.info("sentiment_using_openai_fallback")
        return OpenAISentimentFallback()
    else:
        from .crypto_sentiment import FinoGridSentimentAnalyzer
        log.info("sentiment_using_fingpt_model")
        analyzer = FinoGridSentimentAnalyzer()
        analyzer.load()
        return analyzer
