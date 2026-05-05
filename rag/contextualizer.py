from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import APITimeoutError, OpenAI
from .openai_settings import get_openai_api_key, get_openai_base_url


@dataclass
class ContextualizationConfig:
    model: str
    max_context_chars: int = 4000
    request_timeout_seconds: float = 60.0
    max_retries: int = 2
    max_workers: int = 4
    task_queue_size: int = 16

    @classmethod
    def from_env(cls) -> "ContextualizationConfig":
        load_dotenv()
        api_key = get_openai_api_key()
        base_url = get_openai_base_url()
        model = os.getenv("OPENAI_CONTEXT_MODEL", "").strip() or os.getenv("OPENAI_CHAT_MODEL", "").strip()
        max_context_chars = int(os.getenv("OPENAI_CONTEXT_MAX_CHARS", "4000"))
        request_timeout_seconds = float(os.getenv("OPENAI_CONTEXT_TIMEOUT_SECONDS", "60"))
        max_retries = int(os.getenv("OPENAI_CONTEXT_MAX_RETRIES", "2"))
        max_workers = int(os.getenv("OPENAI_CONTEXT_MAX_WORKERS", "4"))
        task_queue_size = int(os.getenv("OPENAI_CONTEXT_TASK_QUEUE_SIZE", "16"))
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required when contextual chunking is enabled")
        if not model:
            raise ValueError("OPENAI_CONTEXT_MODEL (or OPENAI_CHAT_MODEL) is required when contextual chunking is enabled")
        if max_context_chars <= 0:
            raise ValueError("OPENAI_CONTEXT_MAX_CHARS must be > 0")
        if request_timeout_seconds <= 0:
            raise ValueError("OPENAI_CONTEXT_TIMEOUT_SECONDS must be > 0")
        if max_retries < 0:
            raise ValueError("OPENAI_CONTEXT_MAX_RETRIES must be >= 0")
        if max_workers <= 0:
            raise ValueError("OPENAI_CONTEXT_MAX_WORKERS must be > 0")
        if task_queue_size <= 0:
            raise ValueError("OPENAI_CONTEXT_TASK_QUEUE_SIZE must be > 0")
        return cls(
            model=model,
            max_context_chars=max_context_chars,
            request_timeout_seconds=request_timeout_seconds,
            max_retries=max_retries,
            max_workers=max_workers,
            task_queue_size=task_queue_size,
        )


class ChunkContextualizer:
    def __init__(self, config: ContextualizationConfig):
        self.config = config
        self.client = OpenAI(
            api_key=get_openai_api_key(),
            base_url=get_openai_base_url(),
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

    def contextualize_many(self, *, full_text: str, chunk_texts: list[str]) -> list[str | None]:
        if not chunk_texts:
            return []

        worker_count = min(self.config.max_workers, len(chunk_texts))
        queue_limit = max(self.config.task_queue_size, worker_count)
        results: list[str | None] = [None] * len(chunk_texts)

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            pending: dict[Future[str], int] = {}
            next_index = 0

            while next_index < len(chunk_texts) or pending:
                while next_index < len(chunk_texts) and len(pending) < queue_limit:
                    future = executor.submit(self.contextualize, full_text=full_text, chunk_text=chunk_texts[next_index])
                    pending[future] = next_index
                    next_index += 1

                done, _ = wait(tuple(pending.keys()), return_when=FIRST_COMPLETED)
                for future in done:
                    idx = pending.pop(future)
                    try:
                        value = (future.result() or "").strip()
                        results[idx] = value if value else None
                    except Exception:
                        results[idx] = None

        return results

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
