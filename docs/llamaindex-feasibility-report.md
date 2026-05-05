# LlamaIndex 可行性评估与重构方案

> 目标：评估是否可以用 LlamaIndex 重构当前 `rag_demo`，并给出可执行的重构路径（仅方案，不改现有业务代码）。

## 1. 结论（TL;DR）

结论：**可行，且建议重构**。

原因：你们当前已经自行实现了较多“RAG编排层能力”（查询路由、子问题分解、多路检索、分数阈值、进度回调、异步任务）。这些能力在 LlamaIndex 中已有较成熟的抽象：
- 查询编排：`SubQuestionQueryEngine`、`ToolRetrieverRouterQueryEngine`
- 检索能力：`VectorIndexRetriever`、`BM25Retriever`、`QueryFusionRetriever`（示例）
- 分数过滤：`SimilarityPostprocessor(similarity_cutoff=...)`
- 文档入库：`IngestionPipeline`（含缓存、异步）
- 可观测性：`CallbackManager` / Workflows instrumentation
- 工作流编排：`Workflow`（事件驱动、异步优先）

因此，用 LlamaIndex 重构可以显著减少自研编排代码，降低维护成本，并提升可扩展性。

---

## 2. 现状与目标对齐

你们当前关键需求（按最近讨论）与 LlamaIndex 能力映射：

1. 多问题类型路由（fact/scan/summary/semantic/steps）
- 可用：Router Query Engine（推荐 ToolRetrieverRouter 路线）

2. 语义复杂问题拆解 + 并行子检索 + 汇总
- 可用：`SubQuestionQueryEngine`（支持将复杂问题拆成子问题并汇总）

3. 混合检索（语义 + 关键词）
- 可用：向量检索 + BM25 检索 + Fusion（官方示例有 QueryFusionRetriever）

4. 最低分数过滤
- 可用：`SimilarityPostprocessor` 的 `similarity_cutoff`

5. 文档结构化切分（Markdown标题/表格）
- 可用：`MarkdownNodeParser`、`MarkdownElementNodeParser`

6. 入库流水线、缓存、异步
- 可用：`IngestionPipeline`，支持 `run/arun` 与缓存

7. 进度/可观测性
- 可用：Callbacks（`CHUNKING/NODE_PARSING/EMBEDDING/LLM/QUERY/...`）
- 可用：Workflows + instrumentation（事件级可观测）

8. QA对生成
- 可用：评测/数据生成模块（`DatasetGenerator`，并提示已推荐新替代）
- 实际生产可用自定义 transform（更可控）

---

## 3. 官方能力证据（对应需求）

1. 子问题分解与汇总
- SubQuestionQueryEngine（复杂查询拆解为子问题后执行并汇总）
- https://docs.llamaindex.ai/en/v0.10.33/api_reference/query_engine/sub_question/

2. 路由查询
- ToolRetrieverRouterQueryEngine
- https://docs.llamaindex.ai/en/stable/api_reference/query_engine/tool_retriever_router/

3. 检索基础
- RetrieverQueryEngine
- https://docs.llamaindex.ai/en/stable/api_reference/query_engine/retriever/

4. 向量检索参数（top_k、filters、doc_ids、hybrid alpha）
- VectorIndexRetriever
- https://docs.llamaindex.ai/en/stable/api_reference/retrievers/vector/

5. BM25检索
- BM25Retriever
- https://docs.llamaindex.ai/en/latest/api_reference/retrievers/bm25/

6. 融合检索示例（语义+关键词）
- Relative/Distribution Score Fusion 示例（QueryFusionRetriever）
- https://docs.llamaindex.ai/en/v0.10.33/examples/retrievers/relative_score_dist_fusion/

7. 最低分过滤
- SimilarityPostprocessor(similarity_cutoff)
- https://docs.llamaindex.ai/en/latest/api_reference/postprocessor/similarity/

8. Markdown结构解析
- MarkdownNodeParser
- https://docs.llamaindex.ai/en/stable/api_reference/node_parsers/markdown/
- MarkdownElementNodeParser（含表格元素）
- https://docs.llamaindex.ai/en/stable/api_reference/node_parsers/markdown_element/

9. 入库流水线（缓存、异步）
- IngestionPipeline（含 `arun`）
- https://docs.llamaindex.ai/en/stable/module_guides/loading/ingestion_pipeline/

10. 回调与可观测事件
- Callbacks（事件类型）
- https://docs.llamaindex.ai/en/stable/module_guides/observability/callbacks/
- Event types（QueryStart/End、RetrievalStart/End 等）
- https://docs.llamaindex.ai/en/stable/api_reference/instrumentation/event_types/

11. Workflow（事件驱动 + async）
- https://docs.llamaindex.ai/en/stable/module_guides/workflow/

12. QA/问答数据生成
- Dataset generation API（注：页面提示旧类已建议迁移）
- https://docs.llamaindex.ai/en/stable/api_reference/evaluation/dataset_generation/

---

## 4. 可行性判断

### 4.1 技术可行性
高。你们当前核心需求都能在 LlamaIndex 找到对应抽象；少量定制逻辑（比如 scan 的正则生成与全文匹配）也能通过自定义 Retriever/Tool 接入。

### 4.2 改造复杂度
中高。原因不是“框架不足”，而是你们现有系统已经有一层定制编排：
- Web 异步任务队列
- 定制元数据透出（regexes/sub_questions/progress）
- 自定义 scan 逻辑

这些可以保留外层接口不变，内部替换为 LlamaIndex 组件，分阶段迁移风险可控。

### 4.3 风险点
1. 版本差异：LlamaIndex 文档存在 `stable` 与 `v0.10/v0.12` 路径差异，需固定版本。
2. 功能命名演进：如某些旧API标注 deprecated，需要避开旧路径。
3. 现有行为一致性：要做回归，保证你们前端看到的阶段、元数据结构不被破坏。

---

## 5. 重构总体策略

建议采用：**外部接口保持不变，内部引擎逐步替换**。

- 保留当前 Flask API、任务轮询协议、前端字段。
- 在 `rag_pipeline` 内部改为 LlamaIndex 组件。
- 逐步替换，避免一次性大爆炸。

---

## 6. 目标架构（LlamaIndex版）

1. Ingestion 层
- `IngestionPipeline(transformations=[...])`
- NodeParser：`MarkdownNodeParser` 或 `MarkdownElementNodeParser`
- Embedding：OpenAI兼容 embedding
- 可选 transform：QA 对生成（自定义）

2. Retrieval 层
- Fact：`VectorIndexRetriever`
- Scan：自定义 `RegexScanRetriever`（保留你们 regex 特色）
- Hybrid：`VectorIndexRetriever + BM25Retriever + Fusion`
- Score Cutoff：`SimilarityPostprocessor`

3. Query Orchestration 层
- Router：ToolRetrieverRouter（或 Router Query Engine）
- Semantic复杂问题：`SubQuestionQueryEngine`

4. Observability 层
- `CallbackManager` + 事件映射
- 将 LlamaIndex 事件映射回你们的 progress stage（routing/retrieving/decomposing/selecting 等）

5. Web 层
- 保持现有 `/api/upload-task`、`/api/query-task`、`/api/tasks/{id}`
- 仅替换任务执行体

---

## 7. 分阶段重构计划

### Phase 0：基线冻结
- 固定 LlamaIndex 版本
- 记录当前回归集（典型问题+预期行为）

### Phase 1：仅替换 Ingestion
- 用 IngestionPipeline 替换当前 ingest
- 保持查询链路不变
- 验证 chunk 数、入库性能、元数据完整性

### Phase 2：替换 Fact/Hybrid 检索
- 引入 VectorIndexRetriever/BM25/Fusion
- 接入 SimilarityPostprocessor 作为 min_score
- 保持 scan/semantic 先不动

### Phase 3：替换 Semantic 路由
- 用 SubQuestionQueryEngine 接管复杂语义问题
- 将子问题列表写入结果元数据（维持前端展示）

### Phase 4：接入 Router
- 用 ToolRetrieverRouter 做问题类型路由
- 与现有 route 元数据对齐

### Phase 5：统一可观测与任务进度
- Callback 事件到你们 progress_cb 的映射
- 验证前端轮询体验无回退

### Phase 6：清理旧实现
- 移除废弃自研逻辑
- 保留必要的自定义插件（regex scan、QA transform）

---

## 8. “能否直接整体重构”结论

可以，但不建议“一步到位”直接切换生产。

建议：
- 用 2~3 周做“并行双轨重构”（新旧引擎可切换）
- 通过 A/B 验证准确率与时延
- 达标后再切主

达标门槛建议：
1. 事实定位准确率不低于现网
2. 语义复杂问题准确率提升（重点）
3. 查询时延可控（P95不显著退化）
4. 进度可观测能力不下降

---

## 9. 对你们项目的建议落点

1. 保留自定义 scan（regex）
- 这是你们业务特色，LlamaIndex做框架底座即可

2. 优先迁移 semantic 分解
- 这是准确率收益最大的模块

3. 先把 min_score 统一成 postprocessor
- 避免散落在各分支做手工过滤

4. QA 对生成功能做成 ingestion transform
- 与主入库流程解耦，便于开关与扩展

---

## 10. 下一步（如果你确认推进）

我可以继续输出第二份文档：
- 《LlamaIndex 重构技术设计稿（面向当前仓库）》
内容会细到：模块替换清单、类映射表、API不变策略、测试清单、回滚策略、工期拆分。

---

## 附录：已核验官方链接（docs.llamaindex.ai）

以下链接均来自 LlamaIndex 官方文档域名：

- SubQuestionQueryEngine（stable）
  - https://docs.llamaindex.ai/en/stable/api_reference/query_engine/sub_question/
- ToolRetrieverRouterQueryEngine（stable）
  - https://docs.llamaindex.ai/en/stable/api_reference/query_engine/tool_retriever_router/
- IngestionPipeline（stable）
  - https://docs.llamaindex.ai/en/stable/module_guides/loading/ingestion_pipeline/
- SimilarityPostprocessor（stable/latest）
  - https://docs.llamaindex.ai/en/latest/api_reference/postprocessor/similarity/
- Retriever Query Engine（stable）
  - https://docs.llamaindex.ai/en/stable/api_reference/query_engine/retriever/
- VectorIndexRetriever（stable）
  - https://docs.llamaindex.ai/en/stable/api_reference/retrievers/vector/
- BM25Retriever（latest）
  - https://docs.llamaindex.ai/en/latest/api_reference/retrievers/bm25/
- MarkdownNodeParser（stable）
  - https://docs.llamaindex.ai/en/stable/api_reference/node_parsers/markdown/
- MarkdownElementNodeParser（stable）
  - https://docs.llamaindex.ai/en/stable/api_reference/node_parsers/markdown_element/
- Callbacks（stable）
  - https://docs.llamaindex.ai/en/stable/module_guides/observability/callbacks/
- Instrumentation Event Types（stable）
  - https://docs.llamaindex.ai/en/stable/api_reference/instrumentation/event_types/
- Workflow（stable）
  - https://docs.llamaindex.ai/en/stable/module_guides/workflow/
- Dataset generation（stable）
  - https://docs.llamaindex.ai/en/stable/api_reference/evaluation/dataset_generation/

注：官方文档存在版本并行（stable/v0.10/v0.12）；重构实施时建议固定单一版本并对齐API。
