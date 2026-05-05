# Minimal RAG Demo

一个最小可运行的 RAG 流程示例，包含：

1. 文档解析（抽象 parser + markdown parser）
2. 文档 chunk（抽象 chunker + character chunker）
3. OpenAI 向量化（支持 .env 配置）
4. 轻量级内嵌向量存储（SQLite）
5. 检索（query -> embedding -> topk）
6. Web 管理页面（上传 markdown 并执行 chunk+embedding）

## 1. 创建并激活 venv

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. 配置 .env

复制 `.env.example` 为 `.env` 并填入：

```env
OPENAI_API_KEY=your_key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CONTEXT_API_KEY=your_key
OPENAI_CONTEXT_BASE_URL=https://api.openai.com/v1
OPENAI_CONTEXT_MODEL=gpt-4.1-mini
OPENAI_CONTEXT_MAX_CHARS=4000
```

## 3. CLI 模式

### ingest

```powershell
python -m rag.main ingest --input-dir data --db-path rag.db --separator "\n\n" --max-length 500 --overlap 50
```

启用 contextual retrieval（为每个 chunk 追加上下文摘要）：

```powershell
python -m rag.main ingest --input-dir data --db-path rag.db --chunker character --with-contextual-summary
```

### query

```powershell
python -m rag.main query --db-path rag.db --text "你的问题" --top-k 3
```

## 4. Web 管理页

```powershell
python -m rag.web
```

浏览器打开：`http://127.0.0.1:5000`

功能：
- 上传 `.md` 文件
- 指定 `separator/max_length/overlap`
- 可勾选“上下文补全”（对每个 chunk 调用大模型补上下文）
- 自动执行 chunk + embedding + 入库
- 查看历史文档列表（文件名、chunk 数、更新时间）

## 5. 测试

```powershell
pytest -q
```
