"""
Integration tests for MiniMax provider.

These tests call the real MiniMax API and require:
  - MINIMAX_API_KEY environment variable set

Run with:
  MINIMAX_API_KEY=your_key pytest finogrid/tests/integration/test_minimax_integration.py -v
"""
import os
import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv("MINIMAX_API_KEY"),
    reason="MINIMAX_API_KEY not set — skipping live integration tests",
)


@pytest.mark.asyncio
async def test_minimax_sentiment_live():
    """Call MiniMax API to score a financial headline and verify the response shape."""
    from finogrid.fingpt_integration.sentiment.minimax_provider import MiniMaxSentimentProvider

    provider = MiniMaxSentimentProvider()
    result = await provider.score("Apple stock surges to all-time high on strong earnings")

    assert "label" in result
    assert result["label"] in ("positive", "negative", "neutral")
    assert "score" in result
    assert result["score"] in (1, 0, -1)
    assert "raw" in result


@pytest.mark.asyncio
async def test_minimax_sentiment_corridor_news_live():
    """Score a batch of corridor news items via the MiniMax API."""
    from finogrid.fingpt_integration.sentiment.minimax_provider import MiniMaxSentimentProvider

    provider = MiniMaxSentimentProvider()
    news = [
        {"headline": "Brazil GDP grows 3%", "summary": "Economy beats expectations"},
        {"headline": "PIX outage nationwide", "summary": "Central bank investigates"},
    ]
    results = await provider.score_corridor_news(news, "BR")

    assert len(results) == 2
    for r in results:
        assert r["corridor"] == "BR"
        assert r["sentiment_label"] in ("positive", "negative", "neutral")
        assert r["sentiment_score"] in (1, 0, -1)


@pytest.mark.asyncio
async def test_minimax_llm_client_live():
    """Call the MiniMax LLM client and verify it returns a non-empty string."""
    from finogrid.fingpt_integration.minimax_llm_client import MiniMaxLLMClient

    client = MiniMaxLLMClient()
    response = await client.chat("What is 2 + 2? Answer with just the number.")

    assert isinstance(response, str)
    assert len(response) > 0
    assert "4" in response


@pytest.mark.asyncio
async def test_minimax_llm_client_with_agent():
    """Verify MiniMaxLLMClient works as an llm_client for InternalSupportAgent."""
    from finogrid.fingpt_integration.minimax_llm_client import MiniMaxLLMClient
    from finogrid.agents.internal_support.agent import InternalSupportAgent

    client = MiniMaxLLMClient()
    agent = InternalSupportAgent(knowledge_base=None, llm_client=client)

    result = await agent.answer("What is Finogrid?")

    assert "answer" in result
    assert isinstance(result["answer"], str)
    assert len(result["answer"]) > 0
