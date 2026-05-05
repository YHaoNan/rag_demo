from __future__ import annotations

import re
from pathlib import Path
from uuid import uuid4

from flask import Flask, abort, redirect, render_template, request, url_for

from .chunker import CharacterDocumentChunker, DocumentChunker, MarkDownParentChildChunker
from .contextualizer import ChunkContextualizer, ContextualizationConfig
from .embeddings import EmbeddingConfig, OpenAIEmbedder
from .parsers import MarkdownParser
from .rag_pipeline import ingest_document, retrieve
from .vector_store import SQLiteVectorStore


def _slugify(filename: str) -> str:
    stem = Path(filename).stem.lower()
    return re.sub(r"[^a-z0-9_-]+", "-", stem).strip("-") or "document"


def _parse_document_ids(raw: str) -> list[int]:
    if not raw.strip():
        return []
    ids: list[int] = []
    for part in raw.split(","):
        p = part.strip()
        if not p:
            continue
        if not p.isdigit():
            continue
        ids.append(int(p))
    return sorted(set(ids))


def _build_chunker(name: str, separator: str, max_length: int, overlap: int, parent_max: int, child_max: int) -> DocumentChunker:
    if name == "character":
        return CharacterDocumentChunker(separator=separator, max_length=max_length, overlap=overlap)
    if name == "markdown_parent_child":
        return MarkDownParentChildChunker(parent_max_chars=parent_max, child_max_chars=child_max)
    raise ValueError(f"Unsupported chunker: {name}")


def create_app() -> Flask:
    app = Flask(__name__)

    root = Path(__file__).resolve().parent.parent
    upload_dir = root / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    db_path = str(root / "rag.db")

    @app.get("/")
    def index():
        with SQLiteVectorStore(db_path) as store:
            docs = store.list_documents()
        return render_template("index.html", docs=docs, query_text="", top_k=3, document_ids_text="", results=[])

    @app.post("/query")
    def query_docs():
        query_text = request.form.get("query_text", "").strip()
        top_k = int(request.form.get("top_k", "3"))
        document_ids_text = request.form.get("document_ids", "").strip()
        doc_ids = _parse_document_ids(document_ids_text)

        if top_k <= 0:
            top_k = 3

        with SQLiteVectorStore(db_path) as store:
            docs = store.list_documents()

        results = []
        if query_text:
            config = EmbeddingConfig.from_env()
            embedder = OpenAIEmbedder(config)
            with SQLiteVectorStore(db_path) as store:
                results = retrieve(query_text, embedder, store, top_k=top_k, document_ids=doc_ids or None)

        return render_template(
            "index.html",
            docs=docs,
            query_text=query_text,
            top_k=top_k,
            document_ids_text=document_ids_text,
            results=results,
        )

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

    @app.post("/upload")
    def upload():
        file = request.files.get("file")
        if file is None or not file.filename:
            return redirect(url_for("index"))
        if not file.filename.lower().endswith(".md"):
            return redirect(url_for("index"))

        chunker_name = request.form.get("chunker", "character")
        separator = request.form.get("separator", "\n\n")
        max_length = int(request.form.get("max_length", "500"))
        overlap = int(request.form.get("overlap", "50"))
        parent_max = int(request.form.get("parent_max_chars", "1200"))
        child_max = int(request.form.get("child_max_chars", "300"))
        with_contextual_summary = request.form.get("with_contextual_summary") == "on"

        safe_name = f"{_slugify(file.filename)}-{uuid4().hex[:8]}.md"
        dest = upload_dir / safe_name
        file.save(dest)

        parser = MarkdownParser()
        chunker = _build_chunker(chunker_name, separator, max_length, overlap, parent_max, child_max)
        config = EmbeddingConfig.from_env()
        embedder = OpenAIEmbedder(config)
        contextualizer = ChunkContextualizer(ContextualizationConfig.from_env()) if with_contextual_summary else None

        with SQLiteVectorStore(db_path) as store:
            ingest_document(dest, parser, chunker, embedder, store, contextualizer=contextualizer)

        return redirect(url_for("index"))

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="127.0.0.1", port=5000, debug=True)
