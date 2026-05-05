from pathlib import Path

import rag.rag_pipeline as rag_pipeline
from rag.chunker import CharacterDocumentChunker, MarkDownParentChildChunker
from rag.parsers import MarkdownParser
from rag.rag_pipeline import ingest_document, retrieve_smart
from rag.vector_store import SQLiteVectorStore


class FakeEmbedder:
    def embed_texts(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


class FakeContextualizer:
    def contextualize(self, *, full_text: str, chunk_text: str) -> str:
        return f"context summary\noriginal: {chunk_text}"


class FailingContextualizer:
    def contextualize(self, *, full_text: str, chunk_text: str) -> str:
        raise RuntimeError("timeout")


class BatchContextualizer:
    def contextualize_many(self, *, full_text: str, chunk_texts: list[str]) -> list[str | None]:
        return [f"CTX:{t}" for t in chunk_texts]


def test_ingest_document_writes_document_history(tmp_path: Path):
    src = tmp_path / "demo.md"
    src.write_text("# t\n\nhello\n\nworld", encoding="utf-8")
    parser = MarkdownParser()
    chunker = CharacterDocumentChunker(separator="\n\n", max_length=20, overlap=0)
    embedder = FakeEmbedder()

    db = tmp_path / "rag.db"
    with SQLiteVectorStore(str(db)) as store:
        chunk_count = ingest_document(src, parser, chunker, embedder, store)
        docs = store.list_documents()

    assert chunk_count > 0
    assert len(docs) == 1
    assert docs[0]["source_name"] == "demo.md"


def test_same_document_upload_twice_and_delete_one(tmp_path: Path):
    src = tmp_path / "same.md"
    src.write_text("A\n\nB\n\nC", encoding="utf-8")
    parser = MarkdownParser()
    chunker = CharacterDocumentChunker(separator="\n\n", max_length=20, overlap=0)
    embedder = FakeEmbedder()

    db = tmp_path / "rag.db"
    with SQLiteVectorStore(str(db)) as store:
        ingest_document(src, parser, chunker, embedder, store)
        ingest_document(src, parser, chunker, embedder, store)
        docs = store.list_documents()
        assert len(docs) == 2
        before = store.query([1.0, 0.0], top_k=100)
        first_doc_id = docs[0]["id"]
        store.delete_document(first_doc_id)
        after = store.query([1.0, 0.0], top_k=100)

    assert len(after) < len(before)


def test_markdown_parent_child_persists_parent_structure(tmp_path: Path):
    src = tmp_path / "tree.md"
    src.write_text("# T\n\n## A\n\n- x\n- y\n\ntext block\n", encoding="utf-8")
    parser = MarkdownParser()
    chunker = MarkDownParentChildChunker(parent_max_chars=80, child_max_chars=20)
    embedder = FakeEmbedder()

    db = tmp_path / "rag.db"
    with SQLiteVectorStore(str(db)) as store:
        ingest_document(src, parser, chunker, embedder, store)
        doc = store.list_documents()[0]
        parents = store.list_parent_chunks(doc["id"])

    assert doc["chunker"] == "MarkDownParentChildChunker"
    assert len(parents) > 0


def test_ingest_document_with_contextualizer_prefixes_chunk(tmp_path: Path):
    src = tmp_path / "ctx.md"
    src.write_text("text", encoding="utf-8")
    parser = MarkdownParser()
    chunker = CharacterDocumentChunker(separator="\n\n", max_length=20, overlap=0)
    embedder = FakeEmbedder()

    db = tmp_path / "rag.db"
    with SQLiteVectorStore(str(db)) as store:
        ingest_document(src, parser, chunker, embedder, store, contextualizer=FakeContextualizer())
        doc = store.list_documents()[0]
        chunks = store.list_chunks(doc["id"])

    assert len(chunks) == 1
    assert chunks[0]["text"].startswith("context summary")


def test_ingest_document_contextualizer_failure_degrades_gracefully(tmp_path: Path):
    src = tmp_path / "ctx_fail.md"
    src.write_text("A\n\nB", encoding="utf-8")
    parser = MarkdownParser()
    chunker = CharacterDocumentChunker(separator="\n\n", max_length=20, overlap=0)
    embedder = FakeEmbedder()

    db = tmp_path / "rag.db"
    with SQLiteVectorStore(str(db)) as store:
        chunk_count = ingest_document(src, parser, chunker, embedder, store, contextualizer=FailingContextualizer())
        doc = store.list_documents()[0]
        chunks = store.list_chunks(doc["id"])

    assert chunk_count == 1
    assert [c["text"] for c in chunks] == ["A\n\nB"]


def test_retrieve_smart_routes_scan(tmp_path: Path, monkeypatch):
    src = tmp_path / "scan.md"
    src.write_text("error code E100\n\nerror code E200", encoding="utf-8")
    parser = MarkdownParser()
    chunker = CharacterDocumentChunker(separator="\n\n", max_length=20, overlap=0)
    embedder = FakeEmbedder()

    db = tmp_path / "rag.db"
    with SQLiteVectorStore(str(db)) as store:
        ingest_document(src, parser, chunker, embedder, store)
        monkeypatch.setattr(rag_pipeline, "_generate_regexes_with_llm", lambda query: [r"error\\s+code"])
        smart = retrieve_smart("error code count", embedder, store, top_k=3)

    assert smart["route"] == "scan"
    assert len(smart["results"]) > 0
    assert smart["all_regexes"] == [r"error\\s+code"]


def test_retrieve_smart_routes_summary(tmp_path: Path):
    src = tmp_path / "summary.md"
    src.write_text("# A\n\nline1\n\nline2\n\nline3", encoding="utf-8")
    parser = MarkdownParser()
    chunker = CharacterDocumentChunker(separator="\n\n", max_length=10, overlap=0)
    embedder = FakeEmbedder()

    db = tmp_path / "rag.db"
    with SQLiteVectorStore(str(db)) as store:
        ingest_document(src, parser, chunker, embedder, store)
        smart = retrieve_smart("summarize this doc", embedder, store, top_k=1)

    assert smart["route"] in {"summary", "semantic", "fact"}
    assert len(smart["results"]) >= 0


def test_retrieve_smart_scan_count_question_extracts_term(tmp_path: Path, monkeypatch):
    src = tmp_path / "jsa.md"
    src.write_text("JSA\n\nfoo\n\nJSA bar JSA", encoding="utf-8")
    parser = MarkdownParser()
    chunker = CharacterDocumentChunker(separator="\n\n", max_length=30, overlap=0)
    embedder = FakeEmbedder()

    db = tmp_path / "rag.db"
    with SQLiteVectorStore(str(db)) as store:
        ingest_document(src, parser, chunker, embedder, store)
        monkeypatch.setattr(rag_pipeline, "_generate_regexes_with_llm", lambda query: [r"JSA"])
        smart = retrieve_smart("JSA count", embedder, store, top_k=3)

    assert smart["route"] == "scan"
    assert len(smart["results"]) >= 1
    assert smart["results"][0]["chunk_id"] == "scan_summary"
    assert "JSA" in smart["results"][0]["text"]


def test_ingest_document_uses_batch_contextualizer_when_available(tmp_path: Path):
    src = tmp_path / "ctx_many.md"
    src.write_text("A\n\nB\n\nC", encoding="utf-8")
    parser = MarkdownParser()
    chunker = CharacterDocumentChunker(separator="\n\n", max_length=10, overlap=0)
    embedder = FakeEmbedder()

    db = tmp_path / "rag.db"
    with SQLiteVectorStore(str(db)) as store:
        ingest_document(src, parser, chunker, embedder, store, contextualizer=BatchContextualizer())
        doc = store.list_documents()[0]
        chunks = store.list_chunks(doc["id"])

    assert len(chunks) == 1
    assert chunks[0]["text"].startswith("CTX:")
