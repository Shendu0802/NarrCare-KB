"""Generic OpenAI-compatible API client."""
from pydantic import BaseModel
from openai import AsyncOpenAI


class LLMConfig(BaseModel):
    """Configuration for an OpenAI-compatible LLM endpoint."""

    base_url: str = "https://api.deepseek.com/v1"
    api_key: str = "sk-6962132f5d174473ae6a078e7c4c320d"
    model: str = "deepseek-v4-flash"
    timeout: int = 90
    max_retries: int = 2


class LLMClient:
    """Async client for OpenAI-compatible chat completion APIs."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = AsyncOpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout,
            max_retries=config.max_retries,
        )

    @staticmethod
    def build_messages(system: str, user: str) -> list[dict]:
        """Build standard messages list from system + user prompts."""
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    async def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        response_format: dict | None = None,
    ) -> str:
        """Send a chat completion request and return the text response."""
        kwargs = dict(
            model=model or self.config.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if response_format is not None:
            kwargs["response_format"] = response_format

        response = await self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    async def chat_with_json(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> dict:
        """Chat completion with JSON mode, returns parsed dict. Retries once on parse failure."""
        import json

        for attempt in range(2):
            text = await self.chat(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                if attempt == 1:
                    raise
        return {}
