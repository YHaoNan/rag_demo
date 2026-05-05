from __future__ import annotations

import argparse

from .chunker import CharacterDocumentChunker, MarkDownParentChildChunker
from .contextualizer import ChunkContextualizer, ContextualizationConfig
from .embeddings import EmbeddingConfig, OpenAIEmbedder
from .parsers import MarkdownParser
from .rag_pipeline import ingest_directory, retrieve_smart
from .vector_store import SQLiteVectorStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal RAG demo")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Parse, chunk, embed and store markdown docs")
    ingest.add_argument("--input-dir", required=True)
    ingest.add_argument("--db-path", required=True)
    ingest.add_argument("--chunker", choices=["character", "markdown_parent_child"], default="character")
    ingest.add_argument("--separator", default="\n\n")
    ingest.add_argument("--max-length", type=int, default=500)
    ingest.add_argument("--overlap", type=int, default=50)
    ingest.add_argument("--parent-max-chars", type=int, default=1200)
    ingest.add_argument("--child-max-chars", type=int, default=300)
    ingest.add_argument("--with-contextual-summary", action="store_true")

    query = sub.add_parser("query", help="Query vector store")
    query.add_argument("--db-path", required=True)
    query.add_argument("--text", required=True)
    uery.add_argument("--top-k", type=int, default=3)

    return parser


def main() -> None:
    args = build_parser().parse_args()

    config = EmbeddingConfig.from_env()
    embedder = OpenAIEmbedder(config)

    if args.command == "ingest":
        parser = MarkdownParser()
        contextualizer = ChunkContextualizer(ContextualizationConfig.from_env()) if args.with_contextual_summary else None
        if args.chunker == "character":
            chunker = CharacterDocumentChunker(separator=args.separator, max_length=args.max_length, overlap=args.overlap)
        else:
            chunker = MarkDownParentChildChunker(parent_max_chars=args.parent_max_chars, child_max_chars=args.child_max_chars)

        with SQLiteVectorStore(args.db_path) as store:
            total = ingest_directory(args.input_dir, parser, chunker, embedder, store, contextualizer=contextualizer)
        print(f"Ingested chunks: {total}")

    elif args.command == "query":
        with SQLiteVectorStore(args.db_path) as store:
            smart = retrieve_smart(args.text, embedder, store, top_k=args.top_k)
            results = smart["results"]
        print(f"route={smart['route']} confidence={smart['confidence']:.2f} reason={smart['reason']}")
        for i, item in enumerate(results, start=1):
            print(f"[{i}] document_id={item['document_id']} source={item['source_name']} chunk={item['chunk_id']} score={item['score']:.4f}")
            print(item["text"])
            print("-" * 40)


if __name__ == "__main__":
    main()
