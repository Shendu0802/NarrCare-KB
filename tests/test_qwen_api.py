"""Test Qwen DashScope embeddings API connectivity."""
import pytest
from openai import OpenAI


@pytest.fixture
def client():
    return OpenAI(
        base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
        api_key='sk-5a858dfd8e3c4e25bdd54d7431346e19',
        timeout=30,
    )


class TestQwenEmbeddingsAPI:
    """Verify Qwen embeddings API is reachable and returns correct dimensions."""

    def test_single_embedding(self, client):
        resp = client.embeddings.create(
            model='text-embedding-v3',
            input=['测试文本'],
            dimensions=1024,
        )
        assert len(resp.data) == 1
        assert len(resp.data[0].embedding) == 1024

    def test_batch_embedding(self, client):
        texts = ['第一条测试文本', '第二条测试文本', '第三条']
        resp = client.embeddings.create(
            model='text-embedding-v3',
            input=texts,
            dimensions=1024,
        )
        assert len(resp.data) == 3
        for item in resp.data:
            assert len(item.embedding) == 1024

    def test_embedding_values_are_normalized(self, client):
        """Qwen embeddings should be roughly normalized (余弦相似度用)."""
        import numpy as np
        resp = client.embeddings.create(
            model='text-embedding-v3',
            input=['安宁疗护中死亡焦虑的叙事护理干预方法'],
            dimensions=1024,
        )
        vec = np.array(resp.data[0].embedding)
        norm = np.linalg.norm(vec)
        # Qwen embeddings may not be pre-normalized, but should be reasonable
        assert 0.9 < norm < 1.1  # Qwen already normalizes to unit length
