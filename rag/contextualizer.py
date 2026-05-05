from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import APITimeoutError, OpenAI


@dataclass
class ContextualizationConfig:
    api_key: str
    base_url: str
    model: str
    max_context_chars: int = 4000
    request_timeout_seconds: float = 60.0
    max_retries: int = 2

    @classmethod
    def from_env(cls) -> "ContextualizationConfig":
        load_dotenv()
        api_key = os.getenv("OPENAI_CONTEXT_API_KEY", "")
        base_url = os.getenv("OPENAI_CONTEXT_BASE_URL", "https://api.openai.com/v1")
        model = os.getenv("OPENAI_CONTEXT_MODEL", "")
        max_context_chars = int(os.getenv("OPENAI_CONTEXT_MAX_CHARS", "4000"))
        request_timeout_seconds = float(os.getenv("OPENAI_CONTEXT_TIMEOUT_SECONDS", "60"))
        max_retries = int(os.getenv("OPENAI_CONTEXT_MAX_RETRIES", "2"))
        if not api_key:
            raise ValueError("OPENAI_CONTEXT_API_KEY is required when contextual chunking is enabled")
        if not model:
            raise ValueError("OPENAI_CONTEXT_MODEL is required when contextual chunking is enabled")
        if max_context_chars <= 0:
            raise ValueError("OPENAI_CONTEXT_MAX_CHARS must be > 0")
        if request_timeout_seconds <= 0:
            raise ValueError("OPENAI_CONTEXT_TIMEOUT_SECONDS must be > 0")
        if max_retries < 0:
            raise ValueError("OPENAI_CONTEXT_MAX_RETRIES must be >= 0")
        return cls(
            api_key=api_key,
            base_url=base_url,
            model=model,
            max_context_chars=max_context_chars,
            request_timeout_seconds=request_timeout_seconds,
            max_retries=max_retries,
        )


class ChunkContextualizer:
    def __init__(self, config: ContextualizationConfig):
        self.config = config
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.request_timeout_seconds,
            max_retries=config.max_retries,
        )

    def contextualize(self, *, full_text: str, chunk_text: str) -> str:
        snippet = self._extract_context_snippet(full_text, chunk_text, self.config.max_context_chars)
        prompt = (
            "你是一个RAG文档分块助手。请基于给定文档上下文，为原始chunk补充简洁上下文摘要。\n"
            "要求：\n"
            "1. 输出两段文本。\n"
            "2. 第一段以“上下文：”开头，2-4句，说明该chunk所在主题/指标/范围。\n"
            "3. 第二段以“原文：”开头，完整保留原始chunk。\n"
            "4. 不要编造上下文，无法判断时明确说“上下文不足”。\n\n"
            f"文档上下文片段：\n{snippet}\n\n"
            f"原始chunk：\n{chunk_text}"
        )
        try:
            res = self.client.chat.completions.create(
                model=self.config.model,
                temperature=0,
                messages=[
                    {"role": "system", "content": "你擅长为RAG chunk补全文档上下文，保持准确简洁。"},
                    {"role": "user", "content": prompt},
                ],
            )
        except APITimeoutError as exc:
            raise RuntimeError(
                f"contextualize timeout: model={self.config.model}, timeout={self.config.request_timeout_seconds}s"
            ) from exc
        return (res.choices[0].message.content or "").strip()

    @staticmethod
    def _extract_context_snippet(full_text: str, chunk_text: str, max_chars: int) -> str:
        if len(full_text) <= max_chars:
            return full_text
        idx = full_text.find(chunk_text)
        if idx < 0:
            return full_text[:max_chars]
        center = idx + len(chunk_text) // 2
        half = max_chars // 2
        start = max(0, center - half)
        end = min(len(full_text), start + max_chars)
        start = max(0, end - max_chars)
        return full_text[start:end]
