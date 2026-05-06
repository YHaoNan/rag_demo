from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass
class LlamaRagConfig:
    chat_api_key: str
    chat_base_url: str
    chat_model: str
    embed_api_key: str
    embed_base_url: str
    embedding_model: str
    persist_dir: str
    upload_dir: str
    chunk_size: int
    chunk_overlap: int
    embed_batch_size: int
    debug: bool
    session_ttl_seconds: int
    task_store_file: str

    @classmethod
    def from_env(cls) -> "LlamaRagConfig":
        load_dotenv()
        chat_api_key = os.getenv("OPENAI_CHAT_API_KEY", os.getenv("OPENAI_API_KEY", "")).strip()
        embed_api_key = os.getenv("OPENAI_EMBED_API_KEY", os.getenv("OPENAI_API_KEY", "")).strip()
        if not chat_api_key:
            raise ValueError("OPENAI_CHAT_API_KEY is required")
        if not embed_api_key:
            raise ValueError("OPENAI_EMBED_API_KEY is required")
        return cls(
            chat_api_key=chat_api_key,
            chat_base_url=os.getenv("OPENAI_CHAT_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).strip(),
            chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini").strip(),
            embed_api_key=embed_api_key,
            embed_base_url=os.getenv("OPENAI_EMBED_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).strip(),
            embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small").strip(),
            persist_dir=os.getenv("LLAMA_RAG_PERSIST_DIR", "rag_llama_storage").strip(),
            upload_dir=os.getenv("LLAMA_RAG_UPLOAD_DIR", "uploads_llama").strip(),
            chunk_size=int(os.getenv("LLAMA_RAG_CHUNK_SIZE", "500")),
            chunk_overlap=int(os.getenv("LLAMA_RAG_CHUNK_OVERLAP", "50")),
            embed_batch_size=int(os.getenv("LLAMA_RAG_EMBED_BATCH_SIZE", "10")),
            debug=os.getenv("LLAMA_RAG_DEBUG", "0").strip() in {"1", "true", "True"},
            session_ttl_seconds=int(os.getenv("LLAMA_RAG_SESSION_TTL_SECONDS", "3600")),
            task_store_file=os.getenv("LLAMA_RAG_TASK_STORE_FILE", "rag_llama_storage/tasks.json").strip(),
        )
