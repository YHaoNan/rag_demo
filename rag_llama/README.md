# rag-llama (Minimal)

Minimal flow only: upload -> parse -> split to nodes -> embedding -> vector index persist.

## Install

```bash
pip install -r requirements.txt
```

## Env

Reuse project root `.env`:
- `OPENAI_CHAT_API_KEY` (fallback: `OPENAI_API_KEY`)
- `OPENAI_CHAT_BASE_URL` (fallback: `OPENAI_BASE_URL`)
- `OPENAI_CHAT_MODEL`
- `OPENAI_EMBED_API_KEY` (fallback: `OPENAI_API_KEY`)
- `OPENAI_EMBED_BASE_URL` (fallback: `OPENAI_BASE_URL`)
- `OPENAI_EMBEDDING_MODEL`

Optional:
- `LLAMA_RAG_PERSIST_DIR` (default: `rag_llama_storage`)
- `LLAMA_RAG_UPLOAD_DIR` (default: `uploads_llama`)
- `LLAMA_RAG_CHUNK_SIZE` (default: `500`)
- `LLAMA_RAG_CHUNK_OVERLAP` (default: `50`)
- `LLAMA_RAG_EMBED_BATCH_SIZE` (default: `10`)

For some OpenAI-compatible providers, embedding batch size must be <= 10.

## Supported readers

- PDF: `PDFReader`
- DOCX: `DocxReader`
- Markdown: `MarkdownReader`

## Supported split modes

- Sentence: `SentenceSplitter`
- Semantic: `SemanticSplitterNodeParser`
- Hierarchical: `HierarchicalNodeParser`
- Markdown: `MarkdownNodeParser`
- Markdown element: `MarkdownElementNodeParser`

## Run

```bash
python -m rag_llama.web
```

Open:
- `http://127.0.0.1:5001`
