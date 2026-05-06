from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any
import asyncio

from llama_index.core import Document
from llama_index.core.indices.keyword_table import KeywordTableIndex
from llama_index.core.indices.property_graph import PropertyGraphIndex
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    MarkdownElementNodeParser,
    MarkdownNodeParser,
    SemanticSplitterNodeParser,
    SentenceSplitter,
)
from llama_index.core.types import PydanticProgramMode
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.readers.file import DocxReader, MarkdownReader, PDFReader

from .config import LlamaRagConfig

logger = logging.getLogger("rag_llama.ingest")


@dataclass
class ParseOptions:
    split_mode: str
    chunk_size: int
    chunk_overlap: int
    semantic_buffer_size: int
    semantic_breakpoint_percentile_threshold: int
    hierarchical_chunk_sizes: list[int]


def _build_embed_model(cfg: LlamaRagConfig) -> OpenAILikeEmbedding:
    return OpenAILikeEmbedding(
        model_name=cfg.embedding_model,
        api_key=cfg.embed_api_key,
        api_base=cfg.embed_base_url,
        embed_batch_size=cfg.embed_batch_size,
    )


def _build_llm(cfg: LlamaRagConfig) -> OpenAILike:
    return OpenAILike(
        model=cfg.chat_model,
        api_key=cfg.chat_api_key,
        api_base=cfg.chat_base_url,
        timeout=120,
        is_chat_model=True,
        is_function_calling_model=False,
        should_use_structured_outputs=False,
        # Avoid tool-calling path (`tool_choice`) for OpenAI-compatible providers
        # that don't support that parameter in chat completions.
        pydantic_program_mode=PydanticProgramMode.LLM,
    )


def _ensure_thread_event_loop() -> None:
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)


def _load_reader(file_path: Path):
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return PDFReader()
    if suffix in {".docx", ".doc"}:
        return DocxReader()
    if suffix in {".md", ".markdown"}:
        return MarkdownReader()
    raise ValueError(f"Unsupported file type: {suffix}")


def _load_documents(file_path: Path) -> list[Document]:
    reader = _load_reader(file_path)
    documents = reader.load_data(file=file_path)
    if not documents:
        raise ValueError("No content parsed from file")
    return documents


def _to_int_list(csv_text: str, default_values: list[int]) -> list[int]:
    raw = [x.strip() for x in csv_text.split(",") if x.strip()]
    if not raw:
        return default_values
    values = [int(x) for x in raw]
    values.sort(reverse=True)
    return values


def _build_nodes(documents: list[Document], cfg: LlamaRagConfig, options: ParseOptions):
    mode = options.split_mode
    if mode == "sentence":
        parser = SentenceSplitter(chunk_size=options.chunk_size, chunk_overlap=options.chunk_overlap)
        return parser.get_nodes_from_documents(documents)

    if mode == "semantic":
        parser = SemanticSplitterNodeParser(
            embed_model=_build_embed_model(cfg),
            buffer_size=options.semantic_buffer_size,
            breakpoint_percentile_threshold=options.semantic_breakpoint_percentile_threshold,
        )
        if cfg.debug:
            _log_semantic_pre_split(documents, options, parser)
        return parser.get_nodes_from_documents(documents)

    if mode == "hierarchical":
        parser = HierarchicalNodeParser.from_defaults(chunk_sizes=options.hierarchical_chunk_sizes)
        return parser.get_nodes_from_documents(documents)

    if mode == "markdown":
        parser = MarkdownNodeParser()
        return parser.get_nodes_from_documents(documents)

    if mode == "markdown_element":
        # MarkdownElementNodeParser does async table-summary jobs internally.
        # Flask worker threads may not have an event loop by default.
        _ensure_thread_event_loop()
        parser = MarkdownElementNodeParser(llm=_build_llm(cfg))
        return parser.get_nodes_from_documents(documents)

    raise ValueError(f"Unsupported split mode: {mode}")


def _relation_map(node: Any) -> dict[str, str]:
    output: dict[str, str] = {}
    relationships = getattr(node, "relationships", None)
    if not relationships:
        return output
    for k, v in relationships.items():
        key = str(k).split(".")[-1]
        target = ""
        if isinstance(v, list) and v:
            target = ",".join([getattr(x, "node_id", "") for x in v if getattr(x, "node_id", "")])
        else:
            target = getattr(v, "node_id", "")
        if target:
            output[key] = target
    return output


def _node_preview(node: Any) -> dict[str, Any]:
    text = node.get_content() if hasattr(node, "get_content") else str(node)
    return {
        "id": getattr(node, "node_id", ""),
        "type": node.__class__.__name__,
        "preview": text[:240],
        "metadata": getattr(node, "metadata", {}) or {},
        "relationships": _relation_map(node),
    }


def _normalize_text(s: str) -> str:
    return " ".join(s.split())


def _log_semantic_pre_split(documents: list[Document], options: ParseOptions, parser: SemanticSplitterNodeParser) -> None:
    logger.info(
        "[semantic] start | buffer_size=%s | breakpoint_percentile_threshold=%s | docs=%s",
        options.semantic_buffer_size,
        options.semantic_breakpoint_percentile_threshold,
        len(documents),
    )
    for d_i, doc in enumerate(documents):
        text = doc.get_content() if hasattr(doc, "get_content") else str(doc.text)
        sentences = parser.sentence_splitter(text)
        logger.info("[semantic] doc=%s sentence_count=%s", d_i, len(sentences))
        for s_i, sent in enumerate(sentences):
            logger.info("[semantic] doc=%s sentence[%s]=%s", d_i, s_i, sent[:160].replace("\n", " "))


def _log_semantic_post_split(documents: list[Document], nodes: list[Any], parser: SemanticSplitterNodeParser) -> None:
    doc_sentences: list[list[str]] = []
    for doc in documents:
        text = doc.get_content() if hasattr(doc, "get_content") else str(doc.text)
        doc_sentences.append(parser.sentence_splitter(text))
    if not doc_sentences:
        return
    all_sentences = doc_sentences[0]
    sent_norms = [_normalize_text(s) for s in all_sentences]
    cursor = 0
    logger.info("[semantic] finished | node_count=%s", len(nodes))
    for n_i, node in enumerate(nodes):
        n_text = node.get_content() if hasattr(node, "get_content") else str(node)
        n_norm = _normalize_text(n_text)
        start = cursor
        end = start
        combined = ""
        while end < len(sent_norms):
            candidate = _normalize_text((combined + " " + all_sentences[end]).strip())
            if candidate and candidate in n_norm:
                combined = candidate
                end += 1
            else:
                if end == start:
                    end += 1
                break
        if end <= start:
            end = min(start + 1, len(sent_norms))
        cursor = end
        logger.info(
            "[semantic] node[%s] id=%s sentence_range=[%s,%s) preview=%s",
            n_i,
            getattr(node, "node_id", ""),
            start,
            end,
            n_text[:180].replace("\n", " "),
        )


def ingest_file(file_path: Path, cfg: LlamaRagConfig, options: ParseOptions) -> dict:
    documents = _load_documents(file_path)
    nodes = _build_nodes(documents, cfg, options)
    if cfg.debug and options.split_mode == "semantic":
        parser = SemanticSplitterNodeParser(
            embed_model=_build_embed_model(cfg),
            buffer_size=options.semantic_buffer_size,
            breakpoint_percentile_threshold=options.semantic_breakpoint_percentile_threshold,
        )
        _log_semantic_post_split(documents, nodes, parser)

    # Vectorization/indexing is intentionally disabled in this stage.
    # We keep only reader + splitting output for parser behavior validation.
    created = False

    return {
        "source_name": file_path.name,
        "split_mode": options.split_mode,
        "document_count": len(documents),
        "chunk_count": len(nodes),
        "index_created": created,
        "persist_dir": "(disabled in split-only mode)",
        "nodes": [_node_preview(n) for n in nodes],
    }


def split_file(file_path: Path, cfg: LlamaRagConfig, options: ParseOptions) -> tuple[list[Any], dict[str, Any]]:
    documents = _load_documents(file_path)
    nodes = _build_nodes(documents, cfg, options)
    if cfg.debug and options.split_mode == "semantic":
        parser = SemanticSplitterNodeParser(
            embed_model=_build_embed_model(cfg),
            buffer_size=options.semantic_buffer_size,
            breakpoint_percentile_threshold=options.semantic_breakpoint_percentile_threshold,
        )
        _log_semantic_post_split(documents, nodes, parser)

    result = {
        "source_name": file_path.name,
        "split_mode": options.split_mode,
        "document_count": len(documents),
        "chunk_count": len(nodes),
        "index_created": False,
        "persist_dir": "(not stored yet)",
        "nodes": [_node_preview(n) for n in nodes],
    }
    return nodes, result


def store_indexes(session_id: str, nodes: list[Any], cfg: LlamaRagConfig, index_types: list[str]) -> dict[str, Any]:
    outcomes: list[dict[str, Any]] = []
    for idx_type in index_types:
        try:
            subdir_path = Path(cfg.persist_dir) / idx_type / session_id
            subdir_path.mkdir(parents=True, exist_ok=True)
            subdir = str(subdir_path)
            if idx_type == "vector":
                index = VectorStoreIndex(nodes=nodes, embed_model=_build_embed_model(cfg))
                index.storage_context.persist(persist_dir=subdir)
            elif idx_type == "keyword":
                index = KeywordTableIndex(nodes=nodes, llm=_build_llm(cfg))
                index.storage_context.persist(persist_dir=subdir)
            elif idx_type == "property_graph":
                index = PropertyGraphIndex(
                    nodes=nodes,
                    llm=_build_llm(cfg),
                    embed_model=_build_embed_model(cfg),
                )
                index.storage_context.persist(persist_dir=subdir)
            else:
                raise ValueError(f"unsupported index type: {idx_type}")
            artifacts = [str(p.relative_to(subdir_path)) for p in subdir_path.rglob("*") if p.is_file()]
            outcomes.append({"index_type": idx_type, "ok": True, "persist_dir": subdir, "artifacts": artifacts, "error": ""})
        except Exception as e:
            logger.exception("[index] failed | type=%s", idx_type)
            outcomes.append({"index_type": idx_type, "ok": False, "persist_dir": "", "artifacts": [], "error": str(e)})
    return {"session_id": session_id, "results": outcomes}


def parse_options_from_form(form: Any, cfg: LlamaRagConfig) -> ParseOptions:
    return ParseOptions(
        split_mode=(form.get("split_mode") or "sentence").strip(),
        chunk_size=int(form.get("chunk_size") or cfg.chunk_size),
        chunk_overlap=int(form.get("chunk_overlap") or cfg.chunk_overlap),
        semantic_buffer_size=int(form.get("semantic_buffer_size") or 1),
        semantic_breakpoint_percentile_threshold=int(form.get("semantic_breakpoint_percentile_threshold") or 95),
        hierarchical_chunk_sizes=_to_int_list(form.get("hierarchical_chunk_sizes") or "2048,512,128", [2048, 512, 128]),
    )
