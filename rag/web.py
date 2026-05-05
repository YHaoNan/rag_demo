from __future__ import annotations

import os
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from uuid import uuid4

from dotenv import load_dotenv
from flask import Flask, abort, jsonify, redirect, render_template, request, url_for
from openai import OpenAI

from .chunker import CharacterDocumentChunker, DocumentChunker, MarkDownParentChildChunker
from .contextualizer import ChunkContextualizer, ContextualizationConfig
from .embeddings import EmbeddingConfig, OpenAIEmbedder
from .parsers import MarkdownParser
from .rag_pipeline import ingest_document, retrieve_smart
from .vector_store import SQLiteVectorStore
from .openai_settings import get_openai_api_key, get_openai_base_url


def _slugify(filename: str) -> str:
    stem = Path(filename).stem.lower()
    return re.sub(r"[^a-z0-9_-]+", "-", stem).strip("-") or "document"


def _parse_document_ids(raw: str) -> list[int]:
    if not raw.strip():
        return []
    ids: list[int] = []
    for part in raw.split(","):
        p = part.strip()
        if p.isdigit():
            ids.append(int(p))
    return sorted(set(ids))


def _build_chunker(name: str, separator: str, max_length: int, overlap: int, parent_max: int, child_max: int) -> DocumentChunker:
    if name == "character":
        return CharacterDocumentChunker(separator=separator, max_length=max_length, overlap=overlap)
    if name == "markdown_parent_child":
        return MarkDownParentChildChunker(parent_max_chars=parent_max, child_max_chars=child_max)
    raise ValueError(f"Unsupported chunker: {name}")


@dataclass
class TaskState:
    id: str
    kind: str
    status: str = "queued"
    stage: str = "queued"
    progress: int = 0
    message: str = "Queued"
    result: dict | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class TaskQueue:
    def __init__(self, max_pending: int = 64):
        self.max_pending = max_pending
        self.tasks: dict[str, TaskState] = {}
        self.pending: deque[tuple[str, Callable]] = deque()
        self.lock = threading.Lock()
        self.worker = threading.Thread(target=self._loop, daemon=True)
        self.worker.start()

    def submit(self, kind: str, fn: Callable) -> TaskState:
        with self.lock:
            if len(self.pending) >= self.max_pending:
                raise RuntimeError(f"task queue is full (max={self.max_pending})")
            task = TaskState(id=uuid4().hex, kind=kind)
            self.tasks[task.id] = task
            self.pending.append((task.id, fn))
            return task

    def get(self, task_id: str) -> TaskState | None:
        with self.lock:
            return self.tasks.get(task_id)

    def _update(self, task_id: str, **kwargs) -> None:
        with self.lock:
            task = self.tasks.get(task_id)
            if task is None:
                return
            for k, v in kwargs.items():
                setattr(task, k, v)
            task.updated_at = time.time()

    def _loop(self) -> None:
        while True:
            pair = None
            with self.lock:
                if self.pending:
                    pair = self.pending.popleft()
            if pair is None:
                time.sleep(0.1)
                continue

            task_id, fn = pair
            self._update(task_id, status="running", stage="started", progress=5, message="Task started")
            try:
                result = fn(lambda stage, progress, message, extra=None: self._progress(task_id, stage, progress, message, extra))
                self._update(task_id, status="completed", stage="completed", progress=100, message="Completed", result=result)
            except Exception as exc:
                self._update(task_id, status="failed", stage="failed", message=str(exc), error=str(exc))

    def _progress(self, task_id: str, stage: str, progress: int, message: str, extra: dict | None = None) -> None:
        payload = {"stage": stage, "progress": progress, "message": message}
        if extra:
            payload.update(extra)
        self._update(task_id, **payload)


def create_app() -> Flask:
    app = Flask(__name__)

    root = Path(__file__).resolve().parent.parent
    upload_dir = root / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    db_path = str(root / "rag.db")
    task_queue = TaskQueue(max_pending=int(os.getenv("OPENAI_WEB_TASK_QUEUE_SIZE", "64")))

    @app.get("/")
    def index():
        with SQLiteVectorStore(db_path) as store:
            docs = store.list_documents()
        return render_template(
            "index.html",
            docs=docs,
            query_text="",
            top_k=3,
            document_ids_text="",
            results=[],
            query_route=None,
            query_confidence=None,
            query_reason=None,
        )

    @app.get("/api/tasks/<task_id>")
    def task_status(task_id: str):
        task = task_queue.get(task_id)
        if task is None:
            return jsonify({"error": "task not found"}), 404
        return jsonify(
            {
                "id": task.id,
                "kind": task.kind,
                "status": task.status,
                "stage": task.stage,
                "progress": task.progress,
                "message": task.message,
                "result": task.result,
                "error": task.error,
            }
        )

    @app.post("/api/upload-task")
    def upload_task():
        file = request.files.get("file")
        if file is None or not file.filename:
            return jsonify({"error": "file is required"}), 400
        if not file.filename.lower().endswith(".md"):
            return jsonify({"error": "only .md is supported"}), 400

        chunker_name = request.form.get("chunker", "character")
        separator = request.form.get("separator", "\n\n")
        max_length = int(request.form.get("max_length", "500"))
        overlap = int(request.form.get("overlap", "50"))
        parent_max = int(request.form.get("parent_max_chars", "1200"))
        child_max = int(request.form.get("child_max_chars", "300"))
        with_contextual_summary = request.form.get("with_contextual_summary") == "on"
        with_qa_pairs = request.form.get("with_qa_pairs") == "on"
        qa_pair_count = int(request.form.get("qa_pair_count", "8"))

        safe_name = f"{_slugify(file.filename)}-{uuid4().hex[:8]}.md"
        dest = upload_dir / safe_name
        file.save(dest)

        def run(push):
            push("uploading", 10, "File uploaded", {"source_name": safe_name})
            parser = MarkdownParser()
            chunker = _build_chunker(chunker_name, separator, max_length, overlap, parent_max, child_max)
            config = EmbeddingConfig.from_env()
            embedder = OpenAIEmbedder(config)
            contextualizer = ChunkContextualizer(ContextualizationConfig.from_env()) if with_contextual_summary else None
            with SQLiteVectorStore(db_path) as store:
                ingest_document(
                    dest,
                    parser,
                    chunker,
                    embedder,
                    store,
                    contextualizer=contextualizer,
                    with_qa_pairs=with_qa_pairs,
                    qa_pair_count=qa_pair_count,
                    progress_cb=lambda stage, meta: push(
                        stage,
                        {
                            "parsing": 20,
                            "chunking": 35,
                            "contextualizing": 55,
                            "generating_qa_pairs": 62,
                            "embedding": 70,
                            "indexing": 90,
                            "completed": 98,
                        }.get(stage, 50),
                        meta.get("message", stage),
                        meta,
                    ),
                )
            return {"source_name": safe_name}

        try:
            task = task_queue.submit("upload", run)
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 429
        return jsonify({"task_id": task.id})

    @app.post("/api/query-task")
    def query_task():
        query_text = request.form.get("query_text", "").strip()
        top_k = int(request.form.get("top_k", "3"))
        min_score = float(request.form.get("min_score", "0"))
        document_ids_text = request.form.get("document_ids", "").strip()
        doc_ids = _parse_document_ids(document_ids_text)
        if not query_text:
            return jsonify({"error": "query_text is required"}), 400
        if top_k <= 0:
            top_k = 3

        def run(push):
            config = EmbeddingConfig.from_env()
            embedder = OpenAIEmbedder(config)
            with SQLiteVectorStore(db_path) as store:
                smart = retrieve_smart(
                    query_text,
                    embedder,
                    store,
                    top_k=top_k,
                    document_ids=doc_ids or None,
                    min_score=min_score,
                    progress_cb=lambda stage, meta: push(
                        stage,
                        {"routing": 25, "routed": 40, "retrieving": 75, "completed": 95}.get(stage, 50),
                        meta.get("message", stage),
                        meta,
                    ),
                )
            return smart

        try:
            task = task_queue.submit("query", run)
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 429
        return jsonify({"task_id": task.id})

    @app.post("/api/answer-from-results")
    def answer_from_results():
        payload = request.get_json(silent=True) or {}
        query_text = str(payload.get("query_text", "")).strip()
        top_k = int(payload.get("top_k", 3))
        results = payload.get("results", [])
        if not query_text:
            return jsonify({"error": "query_text is required"}), 400
        if not isinstance(results, list) or not results:
            return jsonify({"error": "results is required and must be non-empty"}), 400

        load_dotenv()
        api_key = get_openai_api_key()
        base_url = get_openai_base_url()
        model = os.getenv("OPENAI_ANSWER_MODEL", "").strip() or os.getenv("OPENAI_CHAT_MODEL", "").strip() or "gpt-4.1-mini"
        timeout = float(os.getenv("OPENAI_ANSWER_TIMEOUT_SECONDS", "60"))
        retries = int(os.getenv("OPENAI_ANSWER_MAX_RETRIES", "2"))
        if not api_key:
            return jsonify({"error": "No API key configured for answer generation"}), 400

        evidence = []
        for i, r in enumerate(results[: max(top_k, 1)], start=1):
            evidence.append(
                f"[{i}] document_id={r.get('document_id')} source={r.get('source_name')} chunk={r.get('chunk_id')} score={r.get('score')}\n{r.get('text','')}"
            )
        evidence_text = "\n\n".join(evidence)

        client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=retries)
        resp = client.chat.completions.create(
            model=model,
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a RAG answerer. Use ONLY the provided retrieval evidence. "
                        "If evidence is insufficient, explicitly say so. Keep answer concise."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question:\n{query_text}\n\nTop-K Evidence:\n{evidence_text}",
                },
            ],
        )
        answer = (resp.choices[0].message.content or "").strip()
        return jsonify({"answer": answer, "model": model})

    @app.get("/docs/<int:document_id>")
    def doc_detail(document_id: int):
        with SQLiteVectorStore(db_path) as store:
            doc = store.get_document(document_id)
            if doc is None:
                abort(404)
            chunks = store.list_chunks(document_id)
            parents = store.list_parent_chunks(document_id) if doc.get("chunker") == "MarkDownParentChildChunker" else []
        return render_template("doc_detail.html", doc=doc, chunks=chunks, parents=parents)

    @app.post("/docs/<int:document_id>/delete")
    def doc_delete(document_id: int):
        with SQLiteVectorStore(db_path) as store:
            store.delete_document(document_id)
        return redirect(url_for("index"))

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="127.0.0.1", port=5000, debug=True)
