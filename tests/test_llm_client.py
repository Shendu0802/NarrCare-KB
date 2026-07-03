import pytest
from src.llm.client import LLMClient, LLMConfig


class TestLLMConfig:
    def test_defaults(self):
        config = LLMConfig(api_key="test-key")
        assert config.base_url == "https://api.deepseek.com/v1"
        assert config.model == "deepseek-v4-flash"
        assert config.timeout == 90
        assert config.max_retries == 2

    def test_custom(self):
        config = LLMConfig(
            base_url="https://api.openai.com/v1",
            api_key="sk-custom",
            model="gpt-4",
            timeout=30,
            max_retries=3,
        )
        assert config.base_url == "https://api.openai.com/v1"
        assert config.model == "gpt-4"


class TestLLMClient:
    def test_client_creation(self):
        config = LLMConfig(api_key="test-key")
        client = LLMClient(config)
        assert client.config.api_key == "test-key"

    def test_build_messages(self):
        config = LLMConfig(api_key="test-key")
        client = LLMClient(config)
        messages = client.build_messages(
            system="You are a helpful assistant.",
            user="Hello",
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
