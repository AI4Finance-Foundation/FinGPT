"""
MiniMax LLM client for Finogrid agents.

Provides a generic ``await client.chat(prompt)`` interface that any Finogrid
agent can use (InternalSupport, AuditGovernance, etc.).

Uses MiniMax's OpenAI-compatible API so the existing ``openai`` dependency
is reused — no new packages required.
"""
from __future__ import annotations

import os
import structlog
from openai import AsyncOpenAI

log = structlog.get_logger()

MINIMAX_BASE_URL = "https://api.minimax.io/v1"


class MiniMaxLLMClient:
    """
    Async LLM client backed by MiniMax's API.

    Compatible with the ``llm_client`` parameter accepted by all Finogrid
    agents (``InternalSupportAgent``, ``AuditGovernanceAgent``, etc.).

    Usage::

        from finogrid.fingpt_integration.minimax_llm_client import MiniMaxLLMClient

        client = MiniMaxLLMClient()
        agent = InternalSupportAgent(knowledge_base=kb, llm_client=client)
    """

    def __init__(
        self,
        model: str = "MiniMax-M2.7",
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ):
        self.model = model
        # MiniMax requires temperature in (0.0, 1.0]
        self.temperature = max(0.01, min(temperature, 1.0))
        self.max_tokens = max_tokens

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

    async def chat(self, prompt: str) -> str:
        """Send a prompt and return the assistant's reply text."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            log.error("minimax_llm_chat_failed", error=str(e))
            raise
