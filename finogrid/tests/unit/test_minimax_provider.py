"""Unit tests for MiniMax sentiment provider and LLM client."""
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# MiniMaxSentimentProvider tests
# ---------------------------------------------------------------------------

class TestMiniMaxSentimentProvider:
    """Tests for MiniMaxSentimentProvider."""

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"})
    def test_init_with_api_key(self):
        from finogrid.fingpt_integration.sentiment.minimax_provider import MiniMaxSentimentProvider
        provider = MiniMaxSentimentProvider()
        assert provider.model == "MiniMax-M2.7"
        assert provider.client is not None

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"})
    def test_init_custom_model(self):
        from finogrid.fingpt_integration.sentiment.minimax_provider import MiniMaxSentimentProvider
        provider = MiniMaxSentimentProvider(model="MiniMax-M2.7-highspeed")
        assert provider.model == "MiniMax-M2.7-highspeed"

    @patch.dict(os.environ, {}, clear=True)
    def test_init_missing_api_key(self):
        # Remove MINIMAX_API_KEY if set
        os.environ.pop("MINIMAX_API_KEY", None)
        from finogrid.fingpt_integration.sentiment.minimax_provider import MiniMaxSentimentProvider
        with pytest.raises(ValueError, match="MINIMAX_API_KEY"):
            MiniMaxSentimentProvider()

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"})
    def test_load_is_noop(self):
        from finogrid.fingpt_integration.sentiment.minimax_provider import MiniMaxSentimentProvider
        provider = MiniMaxSentimentProvider()
        provider.load()  # Should not raise

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"})
    async def test_score_positive(self):
        from finogrid.fingpt_integration.sentiment.minimax_provider import MiniMaxSentimentProvider
        provider = MiniMaxSentimentProvider()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "positive"

        provider.client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await provider.score("Apple stock surges to all-time high")
        assert result["label"] == "positive"
        assert result["score"] == 1
        assert result["raw"] == "positive"

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"})
    async def test_score_negative(self):
        from finogrid.fingpt_integration.sentiment.minimax_provider import MiniMaxSentimentProvider
        provider = MiniMaxSentimentProvider()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "negative"

        provider.client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await provider.score("Market crashes amid recession fears")
        assert result["label"] == "negative"
        assert result["score"] == -1

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"})
    async def test_score_neutral(self):
        from finogrid.fingpt_integration.sentiment.minimax_provider import MiniMaxSentimentProvider
        provider = MiniMaxSentimentProvider()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "neutral"

        provider.client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await provider.score("Fed holds rates steady as expected")
        assert result["label"] == "neutral"
        assert result["score"] == 0

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"})
    async def test_score_api_error_returns_neutral(self):
        from finogrid.fingpt_integration.sentiment.minimax_provider import MiniMaxSentimentProvider
        provider = MiniMaxSentimentProvider()

        provider.client.chat.completions.create = AsyncMock(side_effect=Exception("API error"))

        result = await provider.score("Some news text")
        assert result["label"] == "neutral"
        assert result["score"] == 0
        assert "error" in result

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"})
    async def test_score_temperature_is_low(self):
        """Verify MiniMax uses near-zero temperature (0.01) for deterministic output."""
        from finogrid.fingpt_integration.sentiment.minimax_provider import MiniMaxSentimentProvider
        provider = MiniMaxSentimentProvider()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "positive"

        provider.client.chat.completions.create = AsyncMock(return_value=mock_response)
        await provider.score("test")

        call_kwargs = provider.client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.01
        assert call_kwargs["max_tokens"] == 5

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"})
    async def test_score_corridor_news(self):
        from finogrid.fingpt_integration.sentiment.minimax_provider import MiniMaxSentimentProvider
        provider = MiniMaxSentimentProvider()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "positive"

        provider.client.chat.completions.create = AsyncMock(return_value=mock_response)

        news = [
            {"headline": "Brazil economy grows", "summary": "GDP up 3%"},
            {"headline": "PIX adoption soars", "summary": "50M new users"},
        ]
        results = await provider.score_corridor_news(news, "BR")

        assert len(results) == 2
        assert results[0]["corridor"] == "BR"
        assert results[0]["sentiment_label"] == "positive"
        assert results[0]["sentiment_score"] == 1
        assert results[0]["headline"] == "Brazil economy grows"

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"})
    async def test_score_truncates_long_text(self):
        from finogrid.fingpt_integration.sentiment.minimax_provider import MiniMaxSentimentProvider
        provider = MiniMaxSentimentProvider()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "neutral"

        provider.client.chat.completions.create = AsyncMock(return_value=mock_response)
        long_text = "x" * 1000
        await provider.score(long_text)

        call_kwargs = provider.client.chat.completions.create.call_args[1]
        prompt = call_kwargs["messages"][0]["content"]
        # The prompt should contain at most 512 chars of the input text
        assert len(prompt) < 1000

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"})
    def test_base_url_is_minimax(self):
        from finogrid.fingpt_integration.sentiment.minimax_provider import MiniMaxSentimentProvider
        provider = MiniMaxSentimentProvider()
        assert str(provider.client.base_url).rstrip("/").endswith("api.minimax.io/v1")


# ---------------------------------------------------------------------------
# MiniMaxLLMClient tests
# ---------------------------------------------------------------------------

class TestMiniMaxLLMClient:
    """Tests for MiniMaxLLMClient."""

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"})
    def test_init_defaults(self):
        from finogrid.fingpt_integration.minimax_llm_client import MiniMaxLLMClient
        client = MiniMaxLLMClient()
        assert client.model == "MiniMax-M2.7"
        assert client.temperature == 0.7
        assert client.max_tokens == 1024

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"})
    def test_init_custom_params(self):
        from finogrid.fingpt_integration.minimax_llm_client import MiniMaxLLMClient
        client = MiniMaxLLMClient(
            model="MiniMax-M2.7-highspeed",
            temperature=0.5,
            max_tokens=2048,
        )
        assert client.model == "MiniMax-M2.7-highspeed"
        assert client.temperature == 0.5
        assert client.max_tokens == 2048

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"})
    def test_temperature_clamped_to_min(self):
        """MiniMax requires temperature > 0; verify clamping to 0.01."""
        from finogrid.fingpt_integration.minimax_llm_client import MiniMaxLLMClient
        client = MiniMaxLLMClient(temperature=0.0)
        assert client.temperature == 0.01

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"})
    def test_temperature_clamped_to_max(self):
        """MiniMax requires temperature <= 1.0; verify clamping."""
        from finogrid.fingpt_integration.minimax_llm_client import MiniMaxLLMClient
        client = MiniMaxLLMClient(temperature=2.0)
        assert client.temperature == 1.0

    @patch.dict(os.environ, {}, clear=True)
    def test_init_missing_api_key(self):
        os.environ.pop("MINIMAX_API_KEY", None)
        from finogrid.fingpt_integration.minimax_llm_client import MiniMaxLLMClient
        with pytest.raises(ValueError, match="MINIMAX_API_KEY"):
            MiniMaxLLMClient()

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"})
    async def test_chat_returns_text(self):
        from finogrid.fingpt_integration.minimax_llm_client import MiniMaxLLMClient
        client = MiniMaxLLMClient()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "This is a test response."

        client.client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await client.chat("Hello, world!")
        assert result == "This is a test response."

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"})
    async def test_chat_strips_whitespace(self):
        from finogrid.fingpt_integration.minimax_llm_client import MiniMaxLLMClient
        client = MiniMaxLLMClient()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "  response with spaces  \n"

        client.client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await client.chat("prompt")
        assert result == "response with spaces"

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"})
    async def test_chat_api_error_raises(self):
        from finogrid.fingpt_integration.minimax_llm_client import MiniMaxLLMClient
        client = MiniMaxLLMClient()

        client.client.chat.completions.create = AsyncMock(side_effect=Exception("API error"))

        with pytest.raises(Exception, match="API error"):
            await client.chat("prompt")

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"})
    async def test_chat_passes_correct_params(self):
        from finogrid.fingpt_integration.minimax_llm_client import MiniMaxLLMClient
        client = MiniMaxLLMClient(model="MiniMax-M2.7-highspeed", temperature=0.3, max_tokens=512)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ok"

        client.client.chat.completions.create = AsyncMock(return_value=mock_response)
        await client.chat("test prompt")

        call_kwargs = client.client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "MiniMax-M2.7-highspeed"
        assert call_kwargs["temperature"] == 0.3
        assert call_kwargs["max_tokens"] == 512
        assert call_kwargs["messages"] == [{"role": "user", "content": "test prompt"}]

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"})
    def test_base_url_is_minimax(self):
        from finogrid.fingpt_integration.minimax_llm_client import MiniMaxLLMClient
        client = MiniMaxLLMClient()
        assert str(client.client.base_url).rstrip("/").endswith("api.minimax.io/v1")


# ---------------------------------------------------------------------------
# Factory function tests
# ---------------------------------------------------------------------------

class TestGetSentimentAnalyzerFactory:
    """Tests for the get_sentiment_analyzer() factory."""

    @patch.dict(os.environ, {
        "MINIMAX_API_KEY": "test-key",
        "FINGPT_LLM_PROVIDER": "minimax",
        "FINGPT_USE_OPENAI_FALLBACK": "true",
    })
    def test_factory_returns_minimax_provider(self):
        # Need to reload modules to pick up env changes
        import importlib
        import finogrid.fingpt_integration
        importlib.reload(finogrid.fingpt_integration)

        from finogrid.fingpt_integration.sentiment.minimax_provider import MiniMaxSentimentProvider
        from finogrid.fingpt_integration.sentiment.openai_fallback import get_sentiment_analyzer
        analyzer = get_sentiment_analyzer()
        assert isinstance(analyzer, MiniMaxSentimentProvider)

    @patch.dict(os.environ, {
        "OPENAI_API_KEY": "test-key",
        "FINGPT_LLM_PROVIDER": "openai",
        "FINGPT_USE_OPENAI_FALLBACK": "true",
    })
    def test_factory_returns_openai_provider(self):
        import importlib
        import finogrid.fingpt_integration
        importlib.reload(finogrid.fingpt_integration)

        from finogrid.fingpt_integration.sentiment.openai_fallback import (
            get_sentiment_analyzer, OpenAISentimentFallback,
        )
        analyzer = get_sentiment_analyzer()
        assert isinstance(analyzer, OpenAISentimentFallback)
