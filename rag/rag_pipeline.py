from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv
from openai import OpenAI

from .chunker import CharacterDocumentChunker, DocumentChunker, MarkDownParentChildChunker
from .contextualizer import ChunkContextualizer
from .embeddings import OpenAIEmbedder
from .openai_settings import get_openai_chat_api_key, get_openai_chat_base_url
from .parsers import DocumentParser
from .query_router import QueryRoute, route_query
from .vector_store import SQLiteVectorStore


def _chunker_meta(chunker: DocumentChunker) -> tuple[str, dict]:
    if isinstance(chunker, CharacterDocumentChunker):
        return ("CharacterDocumentChunker", {"separator": chunker.separator, "max_length": chunker.max_length, "overlap": chunker.overlap})
    if isinstance(chunker, MarkDownParentChildChunker):
        return ("MarkDownParentChildChunker", {"parent_max_chars": chunker.parent_max_chars, "child_max_chars": chunker.child_max_chars})
    return (chunker.__class__.__name__, {})


def ingest_document(
    file_path: Path,
    parser: DocumentParser,
    chunker: DocumentChunker,
    embedder: OpenAIEmbedder,
    store: SQLiteVectorStore,
    contextualizer: ChunkContextualizer | None = None,
    with_qa_pairs: bool = False,
    qa_pair_count: int = 8,
    progress_cb: Callable[[str, dict], None] | None = None,
) -> int:
    if progress_cb:
        progress_cb("parsing", {"message": "Parsing document"})
    doc = parser.parse(file_path)
    chunker_name, chunker_params = _chunker_meta(chunker)
    if contextualizer is not None:
        cfg = getattr(contextualizer, "config", None)
        chunker_params = {
            **chunker_params,
            "with_contextual_summary": True,
            "context_model": getattr(cfg, "model", ""),
            "context_max_chars": getattr(cfg, "max_context_chars", 0),
            "context_max_workers": getattr(cfg, "max_workers", 0),
            "context_task_queue_size": getattr(cfg, "task_queue_size", 0),
        }

    parents = []
    if isinstance(chunker, MarkDownParentChildChunker):
        if progress_cb:
            progress_cb("chunking", {"message": "Chunking with markdown parent-child"})
        hierarchy = chunker.chunk_with_hierarchy(doc)
        chunks = hierarchy.children
        parents = hierarchy.parents
    else:
        if progress_cb:
            progress_cb("chunking", {"message": "Chunking document"})
        chunks = chunker.chunk(doc)

    if contextualizer is not None:
        if progress_cb:
            progress_cb("contextualizing", {"message": "Generating contextual summaries"})
        chunk_texts = [c.text for c in chunks]
        if hasattr(contextualizer, "contextualize_many"):
            context_results = contextualizer.contextualize_many(full_text=doc.text, chunk_texts=chunk_texts)
            for i, context_text in enumerate(context_results):
                if context_text:
                    chunks[i].text = f"{context_text}\n"
        else:
            for c in chunks:
                try:
                    context_text = contextualizer.contextualize(full_text=doc.text, chunk_text=c.text)
                    if context_text:
                        c.text = f"{context_text}\n"
                except Exception:
                    continue

    if with_qa_pairs and chunks:
        if progress_cb:
            progress_cb("generating_qa_pairs", {"message": "Generating QA pairs from full document"})
        qa_pairs = _generate_qa_pairs_with_llm(doc.text, qa_pair_count=qa_pair_count)
        chunk_cls = chunks[0].__class__
        for i, qa in enumerate(qa_pairs):
            chunks.append(
                chunk_cls(
                    doc_id=doc.doc_id,
                    chunk_id=f"{file_path.stem}-qa-{i}",
                    text=f"Q: {qa['question']}\nA: {qa['answer']}",
                )
            )

    if progress_cb:
        progress_cb("embedding", {"message": "Embedding chunks", "chunk_count": len(chunks)})
    vectors = embedder.embed_texts([c.text for c in chunks])

    if progress_cb:
        progress_cb("indexing", {"message": "Writing vectors to index"})
    document_id = store.create_document(source_name=file_path.name, chunk_count=len(chunks), chunker=chunker_name, chunker_params=chunker_params)
    rows = [(c.chunk_id, c.text, v) for c, v in zip(chunks, vectors)]
    store.add_many(document_id=document_id, rows=rows)
    if parents:
        parent_rows = [(p.parent_id, p.text, p.child_ids, i) for i, p in enumerate(parents)]
        store.add_parent_chunks(document_id=document_id, rows=parent_rows)
    if progress_cb:
        progress_cb("completed", {"message": "Ingestion completed", "chunk_count": len(chunks), "document_id": document_id})
    return len(chunks)


def ingest_directory(
    input_dir: str,
    parser: DocumentParser,
    chunker: DocumentChunker,
    embedder: OpenAIEmbedder,
    store: SQLiteVectorStore,
    contextualizer: ChunkContextualizer | None = None,
    with_qa_pairs: bool = False,
    qa_pair_count: int = 8,
) -> int:
    input_path = Path(input_dir)
    md_files = sorted(input_path.glob("*.md"))
    total_chunks = 0
    for file_path in md_files:
        total_chunks += ingest_document(
            file_path,
            parser,
            chunker,
            embedder,
            store,
            contextualizer=contextualizer,
            with_qa_pairs=with_qa_pairs,
            qa_pair_count=qa_pair_count,
        )
    return total_chunks


def retrieve(query: str, embedder: OpenAIEmbedder, store: SQLiteVectorStore, top_k: int = 3, document_ids: list[int] | None = None) -> list[dict]:
    q_vec = embedder.embed_query(query)
    return store.query(query_embedding=q_vec, top_k=top_k, document_ids=document_ids)


def _generate_qa_pairs_with_llm(full_text: str, qa_pair_count: int = 8) -> list[dict]:
    load_dotenv()
    api_key = get_openai_chat_api_key()
    if not api_key:
        return []
    base_url = get_openai_chat_base_url()
    model = os.getenv("OPENAI_QA_MODEL", "").strip() or os.getenv("OPENAI_CHAT_MODEL", "").strip() or "gpt-4.1-mini"
    timeout = float(os.getenv("OPENAI_QA_TIMEOUT_SECONDS", "90"))
    retries = int(os.getenv("OPENAI_QA_MAX_RETRIES", "2"))
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=retries)
    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": 'Generate QA pairs from the document. Return strict JSON: {"qa_pairs":[{"question":"...","answer":"..."}]}.',
            },
            {"role": "user", "content": f"Generate {qa_pair_count} QA pairs from this document:\n\n{full_text}"},
        ],
    )
    content = (resp.choices[0].message.content or "").strip()
    data = json.loads(content)
    out = []
    for item in data.get("qa_pairs", []):
        q = str(item.get("question", "")).strip()
        a = str(item.get("answer", "")).strip()
        if q and a:
            out.append({"question": q, "answer": a})
    return out[:qa_pair_count]


def _dedupe_by_chunk_id(rows: list[dict]) -> list[dict]:
    seen: set[tuple[int, str]] = set()
    out: list[dict] = []
    for row in rows:
        key = (row["document_id"], row["chunk_id"])
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _summary_retrieve(query: str, embedder: OpenAIEmbedder, store: SQLiteVectorStore, top_k: int, document_ids: list[int] | None) -> list[dict]:
    dense = retrieve(query=query, embedder=embedder, store=store, top_k=max(top_k * 3, 6), document_ids=document_ids)
    if not dense:
        return []
    grouped: dict[int, dict] = {}
    for item in dense:
        doc_id = item["document_id"]
        if doc_id not in grouped:
            grouped[doc_id] = {"document_id": doc_id, "source_name": item["source_name"], "chunk_id": "summary", "score": item["score"], "texts": []}
        grouped[doc_id]["texts"].append(item["text"])
        grouped[doc_id]["score"] = max(grouped[doc_id]["score"], item["score"])
    out: list[dict] = []
    for row in grouped.values():
        out.append({"document_id": row["document_id"], "source_name": row["source_name"], "chunk_id": "summary", "score": row["score"], "text": "\n---\n".join(row["texts"][:6])})
    out.sort(key=lambda x: x["score"], reverse=True)
    return out[:top_k]


def _generate_regexes_with_llm(query: str) -> list[str]:
    load_dotenv()
    api_key = get_openai_chat_api_key()
    if not api_key:
        raise ValueError("OPENAI_CHAT_API_KEY is required for regex generation")
    base_url = get_openai_chat_base_url()
    model = os.getenv("OPENAI_REGEX_MODEL", "").strip() or os.getenv("OPENAI_CHAT_MODEL", "").strip() or "gpt-4.1-mini"
    timeout = float(os.getenv("OPENAI_REGEX_TIMEOUT_SECONDS", "30"))
    retries = int(os.getenv("OPENAI_REGEX_MAX_RETRIES", "1"))

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=retries)
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": 'Generate 1-5 regular expressions for text search from user query. Return strict JSON: {"regexes":["..."]}.'},
            {"role": "user", "content": query},
        ],
    )
    content = (resp.choices[0].message.content or "").strip()
    data = json.loads(content)
    regexes = [str(x).strip() for x in data.get("regexes", []) if str(x).strip()]
    return regexes[:5]


def _collect_full_text(store: SQLiteVectorStore, document_ids: list[int] | None) -> str:
    docs = store.list_documents()
    target_ids = set(document_ids) if document_ids else {d["id"] for d in docs}
    parts: list[str] = []
    for d in docs:
        if d["id"] not in target_ids:
            continue
        chunks = store.list_chunks(d["id"])
        if chunks:
            parts.append("\n".join(c["text"] for c in chunks))
    return "\n\n".join(parts)


def _keyword_from_regex(pattern: str) -> str:
    words = re.findall(r"[\u4e00-\u9fffA-Za-z0-9_.-]{2,}", pattern or "")
    if words:
        return words[0]
    return pattern.replace("\\", "")


def _scan_retrieve(
    query: str,
    store: SQLiteVectorStore,
    top_k: int,
    document_ids: list[int] | None,
    progress_cb: Callable[[str, dict], None] | None = None,
) -> tuple[list[dict], list[str]]:
    if progress_cb:
        progress_cb("generating_regex", {"message": "Generating regex with LLM"})
    regexes = _generate_regexes_with_llm(query)
    if not regexes:
        return ([], [])

    if progress_cb:
        progress_cb("matching with regex: " + ", ".join(regexes), {"message": "Running regex matching", "all_regexes": regexes})

    full_text = _collect_full_text(store, document_ids)
    total_count = 0
    for pattern in regexes:
        try:
            total_count += len(re.findall(pattern, full_text, flags=re.IGNORECASE | re.MULTILINE))
        except re.error:
            continue

    # Keep current chunk return behavior: still return chunk hits from store search.
    token = _keyword_from_regex(regexes[0])
    hits = store.keyword_search(query=token, top_k=max(top_k * 6, 20), document_ids=document_ids)
    if not hits:
        return ([], regexes)

    summary = {
        "document_id": hits[0]["document_id"],
        "source_name": hits[0]["source_name"],
        "chunk_id": "scan_summary",
        "score": float(total_count),
        "text": f"Regex matching completed. regexes={regexes}, matched_count={total_count}, hit_chunks={len(hits)}.",
    }
    return ([summary] + hits[: max(top_k - 1, 0)], regexes)


def _steps_retrieve(query: str, embedder: OpenAIEmbedder, store: SQLiteVectorStore, top_k: int, document_ids: list[int] | None) -> list[dict]:
    rows = retrieve(query=query, embedder=embedder, store=store, top_k=max(top_k * 4, 12), document_ids=document_ids)
    rows = _dedupe_by_chunk_id(rows)
    rows.sort(key=lambda x: x["chunk_id"])
    return rows[:top_k]


def _semantic_retrieve(query: str, embedder: OpenAIEmbedder, store: SQLiteVectorStore, top_k: int, document_ids: list[int] | None) -> list[dict]:
    dense = retrieve(query=query, embedder=embedder, store=store, top_k=max(top_k * 3, 8), document_ids=document_ids)
    kw = store.keyword_search(query=query, top_k=max(top_k * 3, 8), document_ids=document_ids)
    return _dedupe_by_chunk_id(dense + kw)[:top_k]


def _decompose_semantic_query_with_llm(query: str) -> list[str]:
    load_dotenv()
    api_key = get_openai_chat_api_key()
    if not api_key:
        return [query]
    base_url = get_openai_chat_base_url()
    model = os.getenv("OPENAI_SEMANTIC_DECOMPOSE_MODEL", "").strip() or os.getenv("OPENAI_CHAT_MODEL", "").strip() or "gpt-4.1-mini"
    timeout = float(os.getenv("OPENAI_SEMANTIC_DECOMPOSE_TIMEOUT_SECONDS", "60"))
    retries = int(os.getenv("OPENAI_SEMANTIC_DECOMPOSE_MAX_RETRIES", "2"))

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=retries)
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You decompose semantic user questions into factual sub-questions for retrieval. "
                    "Return strict JSON: {\"sub_questions\":[\"...\"]}. "
                    "Keep each item short and concrete."
                ),
            },
            {"role": "user", "content": query},
        ],
    )
    content = (resp.choices[0].message.content or "").strip()
    data = json.loads(content)
    subs = [str(x).strip() for x in data.get("sub_questions", []) if str(x).strip()]
    return subs[:8] if subs else [query]


def _semantic_retrieve_with_decomposition(
    query: str,
    embedder: OpenAIEmbedder,
    store: SQLiteVectorStore,
    top_k: int,
    document_ids: list[int] | None,
    progress_cb: Callable[[str, dict], None] | None = None,
) -> tuple[list[dict], list[str]]:
    if progress_cb:
        progress_cb("decomposing_subquestions", {"message": "Decomposing semantic query into factual sub-questions"})
    sub_questions = _decompose_semantic_query_with_llm(query)

    each_k = max(top_k, 3)
    workers = min(max(len(sub_questions), 1), int(os.getenv("OPENAI_SEMANTIC_MAX_WORKERS", "6")))
    by_subq: list[list[dict]] = [[] for _ in sub_questions]

    db_path = store.db_path

    def _retrieve_one(i: int, sq: str):
        # SQLite connections are thread-bound; open a fresh connection per worker thread.
        with SQLiteVectorStore(db_path) as thread_store:
            rows = retrieve(query=sq, embedder=embedder, store=thread_store, top_k=each_k, document_ids=document_ids)
        return i, rows

    with ThreadPoolExecutor(max_workers=max(workers, 1)) as ex:
        futures = [ex.submit(_retrieve_one, i, sq) for i, sq in enumerate(sub_questions)]
        for f in as_completed(futures):
            i, rows = f.result()
            by_subq[i] = rows

    if progress_cb:
        progress_cb("selecting_topk", {"message": "Selecting final top-k chunks by round-robin over sub-questions", "sub_questions": sub_questions})

    final: list[dict] = []
    seen: set[tuple[int, str]] = set()
    depth = 0
    while len(final) < top_k:
        picked_this_round = 0
        for i in range(len(sub_questions)):
            rows = by_subq[i]
            if depth >= len(rows):
                continue
            row = rows[depth]
            key = (row["document_id"], row["chunk_id"])
            if key in seen:
                continue
            seen.add(key)
            final.append(row)
            picked_this_round += 1
            if len(final) >= top_k:
                break
        if picked_this_round == 0:
            break
        depth += 1

    return final, sub_questions


def retrieve_smart(
    query: str,
    embedder: OpenAIEmbedder,
    store: SQLiteVectorStore,
    top_k: int = 3,
    document_ids: list[int] | None = None,
    min_score: float | None = None,
    progress_cb: Callable[[str, dict], None] | None = None,
) -> dict:
    if progress_cb:
        progress_cb("routing", {"message": "Routing query"})
    route: QueryRoute = route_query(query)
    if progress_cb:
        progress_cb("routed", {"message": "Route selected", "route": route.route, "reason": route.reason, "confidence": route.confidence})

    all_regexes: list[str] = []
    semantic_sub_questions: list[str] = []
    if route.route == "scan":
        if progress_cb:
            progress_cb("retrieving", {"message": "Scanning with regex"})
        results, all_regexes = _scan_retrieve(query=query, store=store, top_k=top_k, document_ids=document_ids, progress_cb=progress_cb)
    elif route.route == "summary":
        if progress_cb:
            progress_cb("retrieving", {"message": "Retrieving summary candidates"})
        results = _summary_retrieve(query=query, embedder=embedder, store=store, top_k=top_k, document_ids=document_ids)
    elif route.route == "steps":
        if progress_cb:
            progress_cb("retrieving", {"message": "Retrieving ordered step chunks"})
        results = _steps_retrieve(query=query, embedder=embedder, store=store, top_k=top_k, document_ids=document_ids)
    elif route.route == "semantic":
        if progress_cb:
            progress_cb("retrieving", {"message": "Retrieving hybrid semantic evidence"})
        results, semantic_sub_questions = _semantic_retrieve_with_decomposition(
            query=query,
            embedder=embedder,
            store=store,
            top_k=top_k,
            document_ids=document_ids,
            progress_cb=progress_cb,
        )
    else:
        if progress_cb:
            progress_cb("retrieving", {"message": "Retrieving factual evidence"})
        results = retrieve(query=query, embedder=embedder, store=store, top_k=top_k, document_ids=document_ids)

    if min_score is not None:
        results = [r for r in results if float(r.get("score", 0.0)) >= float(min_score)]

    if progress_cb:
        progress_cb(
            "completed",
            {
                "message": "Query completed",
                "result_count": len(results),
                "all_regexes": all_regexes,
                "sub_questions": semantic_sub_questions,
            },
        )
    return {
        "route": route.route,
        "confidence": route.confidence,
        "reason": route.reason,
        "results": results,
        "all_regexes": all_regexes,
        "sub_questions": semantic_sub_questions,
    }
