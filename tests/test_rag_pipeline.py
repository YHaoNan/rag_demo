from pathlib import Path

from rag.chunker import CharacterDocumentChunker, MarkDownParentChildChunker
from rag.parsers import MarkdownParser
from rag.rag_pipeline import ingest_document
from rag.vector_store import SQLiteVectorStore


class FakeEmbedder:
    def embed_texts(self, texts):
        return [[1.0, 0.0] for _ in texts]


class FakeContextualizer:
    def contextualize(self, *, full_text: str, chunk_text: str) -> str:
        return f"上下文：测试上下文\n原文：{chunk_text}"


class FailingContextualizer:
    def contextualize(self, *, full_text: str, chunk_text: str) -> str:
        raise RuntimeError("timeout")


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
    assert docs[0]["chunk_count"] == chunk_count
    assert docs[0]["chunker"] == "CharacterDocumentChunker"


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
    remaining_doc_ids = {r["document_id"] for r in after}
    assert first_doc_id not in remaining_doc_ids
    assert len(remaining_doc_ids) == 1


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
    assert len(parents[0]["child_chunk_ids"]) > 0


def test_ingest_document_with_contextualizer_prefixes_chunk(tmp_path: Path):
    src = tmp_path / "ctx.md"
    src.write_text("该指标不得低于95%。", encoding="utf-8")

    parser = MarkdownParser()
    chunker = CharacterDocumentChunker(separator="\n\n", max_length=20, overlap=0)
    embedder = FakeEmbedder()
    contextualizer = FakeContextualizer()

    db = tmp_path / "rag.db"
    with SQLiteVectorStore(str(db)) as store:
        ingest_document(src, parser, chunker, embedder, store, contextualizer=contextualizer)
        doc = store.list_documents()[0]
        chunks = store.list_chunks(doc["id"])

    assert len(chunks) == 1
    assert chunks[0]["text"].startswith("上下文：测试上下文\n原文：该指标不得低于95%。")


def test_ingest_document_contextualizer_failure_degrades_gracefully(tmp_path: Path):
    src = tmp_path / "ctx_fail.md"
    src.write_text("A\n\nB", encoding="utf-8")

    parser = MarkdownParser()
    chunker = CharacterDocumentChunker(separator="\n\n", max_length=20, overlap=0)
    embedder = FakeEmbedder()
    contextualizer = FailingContextualizer()

    db = tmp_path / "rag.db"
    with SQLiteVectorStore(str(db)) as store:
        chunk_count = ingest_document(src, parser, chunker, embedder, store, contextualizer=contextualizer)
        doc = store.list_documents()[0]
        chunks = store.list_chunks(doc["id"])

    assert chunk_count == 1
    assert [c["text"] for c in chunks] == ["A\n\nB"]
