from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI


@dataclass
class EmbeddingConfig:
    api_key: str
    base_url: str
    model: str

    @classmethod
    def from_env(cls) -> "EmbeddingConfig":
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY", "")
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required in environment or .env")
        return cls(api_key=api_key, base_url=base_url, model=model)


class OpenAIEmbedder:
    def __init__(self, config: EmbeddingConfig, batch_size: int = 10):
        self.config = config
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        self.batch_size = batch_size
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            result = self.client.embeddings.create(model=self.config.model, input=batch)
            all_embeddings.extend(item.embedding for item in result.data)
        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]
