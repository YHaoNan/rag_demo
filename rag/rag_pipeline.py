from __future__ import annotations

from pathlib import Path

from .chunker import CharacterDocumentChunker, DocumentChunker, MarkDownParentChildChunker
from .contextualizer import ChunkContextualizer
from .embeddings import OpenAIEmbedder
from .parsers import DocumentParser
from .vector_store import SQLiteVectorStore


def _chunker_meta(chunker: DocumentChunker) -> tuple[str, dict]:
    if isinstance(chunker, CharacterDocumentChunker):
        return (
            "CharacterDocumentChunker",
            {
                "separator": chunker.separator,
                "max_length": chunker.max_length,
                "overlap": chunker.overlap,
            },
        )
    if isinstance(chunker, MarkDownParentChildChunker):
        return (
            "MarkDownParentChildChunker",
            {
                "parent_max_chars": chunker.parent_max_chars,
                "child_max_chars": chunker.child_max_chars,
            },
        )
    return (chunker.__class__.__name__, {})


def ingest_document(
    file_path: Path,
    parser: DocumentParser,
    chunker: DocumentChunker,
    embedder: OpenAIEmbedder,
    store: SQLiteVectorStore,
    contextualizer: ChunkContextualizer | None = None,
) -> int:
    doc = parser.parse(file_path)
    chunker_name, chunker_params = _chunker_meta(chunker)
    if contextualizer is not None:
        cfg = getattr(contextualizer, "config", None)
        chunker_params = {
            **chunker_params,
            "with_contextual_summary": True,
            "context_model": getattr(cfg, "model", ""),
            "context_max_chars": getattr(cfg, "max_context_chars", 0),
        }
    parents = []
    if isinstance(chunker, MarkDownParentChildChunker):
        hierarchy = chunker.chunk_with_hierarchy(doc)
        chunks = hierarchy.children
        parents = hierarchy.parents
    else:
        chunks = chunker.chunk(doc)

    if contextualizer is not None:
        for c in chunks:
            try:
                context_text = contextualizer.contextualize(full_text=doc.text, chunk_text=c.text)
                if context_text:
                    c.text = f"{context_text}\n"
            except Exception:
                # Degrade gracefully when contextualization fails for a chunk.
                continue

    vectors = embedder.embed_texts([c.text for c in chunks])

    document_id = store.create_document(
        source_name=file_path.name,
        chunk_count=len(chunks),
        chunker=chunker_name,
        chunker_params=chunker_params,
    )

    rows = [(c.chunk_id, c.text, v) for c, v in zip(chunks, vectors)]
    store.add_many(document_id=document_id, rows=rows)
    if parents:
        parent_rows = [(p.parent_id, p.text, p.child_ids, i) for i, p in enumerate(parents)]
        store.add_parent_chunks(document_id=document_id, rows=parent_rows)
    return len(chunks)


def ingest_directory(
    input_dir: str,
    parser: DocumentParser,
    chunker: DocumentChunker,
    embedder: OpenAIEmbedder,
    store: SQLiteVectorStore,
    contextualizer: ChunkContextualizer | None = None,
) -> int:
    input_path = Path(input_dir)
    md_files = sorted(input_path.glob("*.md"))
    total_chunks = 0

    for file_path in md_files:
        total_chunks += ingest_document(file_path, parser, chunker, embedder, store, contextualizer=contextualizer)

    return total_chunks


def retrieve(
    query: str,
    embedder: OpenAIEmbedder,
    store: SQLiteVectorStore,
    top_k: int = 3,
    document_ids: list[int] | None = None,
) -> list[dict]:
    q_vec = embedder.embed_query(query)
    return store.query(query_embedding=q_vec, top_k=top_k, document_ids=document_ids)
