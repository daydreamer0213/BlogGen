# BlogGen 设计文档

## 系统概述

BlogGen 是一个多 Agent 协作的博客生成系统。用户输入学习话题后，5 个 AI Agent 通过 LangGraph 状态机协作生成一篇结构严谨、适合目标读者水平的技术博客。关键节点设有 HITL（人在回路）检查点，让用户在推进前确认中间产物。

### 管线流程

```
用户输入 → 需求对齐 → 知识树构建 → 章节规划 → 撰写(并行) → 审查(并行)
              HITL ✓      HITL ✓       HITL ✓                  HITL ✓
```

每个 HITL 检查点，Streamlit 界面展示当前 Agent 的输出，等待用户确认后再继续。

---

## 1. 架构决策

### 1.1 为什么用 LangGraph 而不是自己写编排

- **内置检查点。** SqliteSaver 提供持久化的暂停/恢复能力，HITL 不需要自己实现状态序列化。
- **Send() 实现并行。** LangGraph 1.x 的 `Send()` 可以从条件边派发并行章节执行，通过 reducer 合并结果——不需要手动管理线程池。
- **条件路由显式化。** `add_conditional_edges()` 让每个节点根据状态决定下一步，路由逻辑清晰可追溯。
- **显式状态 Schema。** BlogGenState TypedDict 配合 `Annotated[list, add]` reducer，状态变更路径一目了然。

代价：多了一个依赖。但手写状态机+线程池的代码量远大于此，且正确性更难保证。

### 1.2 为什么用 DeepSeek 作为 LLM 后端

- **中文能力强。** BGE 中文 Embedding（bge-large-zh-v1.5）+ DeepSeek 原生中文支持，适合目标用户群。
- **双模型策略。** 复杂推理用 V4 Pro（需求对齐、审查），大批量生成用 Flash（撰写）。Flash 成本约 Pro 的 1/12，长文生成效果尚可，当前 MVP 阶段先用 Flash，后续根据实测效果调整。
- **兼容 OpenAI API。** ChatOpenAI 客户端直接可用，不需要厂商专用 SDK。

### 1.3 为什么用 Streamlit

- **纯 Python。** 所有 Agent 都是 Python，UI 也用 Python 避免了语言边界。不需要前端构建工具链。
- **会话状态模型。** `st.session_state` 天然映射到 BlogGenSession 的生命周期——一个浏览器标签页一个会话。
- **HITL 体验。** Streamlit 的回调模式（`on_click`）+ `st.rerun()` 实现了响应式审批流程，不需要 WebSocket。

### 1.4 为什么用 Pydantic

- **边界校验。** 每个 Agent 间的数据交接（LearnerProfile、KnowledgeTree、ChapterPlan、ReviewResult）都经过 `validate_or_raise()`。
- **模糊输入规范化。** `normalize_level_str()` 能处理中文水平描述词（"小白"→beginner，"精通"→advanced），LLM 不需要输出精确字符串。
- **Schema 漂移检测。** Agent 输出格式变化时，Pydantic 立即报错，不会把脏数据传到下游。

---

## 2. Agent 设计

### 2.1 需求对齐（Agent 1）

**职责**：通过多轮对话收集需求，提取结构化的 LearnerProfile。

**设计理由**：
- `needs_alignment` 阶段会循环——图谱在所需字段（domain、level、goal）齐全之前一直路由回同一节点。
- 这是一个天然的 HITL 检查点：每次 LLM 回复后节点中断，用户可以继续聊天或确认。
- 每次 LLM 回复都调用 `extract_json()`，缺少必填字段则触发追问。

**边界处理**：
- 空聊天记录 → 停留在 needs_alignment（UI 显示问候语）
- 信息不完整 → LLM 定向追问（一次只问 1-2 个缺失字段）
- 模糊水平描述 → 由 `normalize_level_str()` 规范化

### 2.2 知识树构建（Agent 2）

**职责**：研究学习领域，输出按学习顺序排列的知识点列表。

**设计理由**：
- 用 **Flash 模型**——知识树是结构化的，不要求创造力，速度更重要。
- 可调用网络搜索（`tavily_search`）和本地知识库（`query_vector_store`）。
- 输出格式刻意设计得简单：`# 领域名\n- 知识点1\n- 知识点2`。格式越简单，解析失败越少。

**重试逻辑**：
- 第一次：标准工具调用流程
- 校验失败 → 用显式格式指令重试
- 再次失败 → 接受已解析到的知识点（优雅降级，不阻塞管线）

### 2.3 章节规划（Agent 3）

**职责**：将知识点归组为章节，再按字数预算拆分为一篇或多篇博客。

**关键设计决策——规划师不编叙事**：
ChapterPlanner 只管分组。叙事结构（核心问题、类比、代码示例）全部交给 Writer。这个分离避免了规划师替写作者做创意决策，导致输出千篇一律。

**拆分算法**（`split_chapters_by_budget`）：
- 在章节边界处递归二分
- 每篇预算 = `每章最大字数 × 5`
- 末尾碎片（<30% 预算）合并到前一篇
- 以章节边界为切分点（自然的语义断点）

### 2.4 撰写（Agent 4）

**职责**：独立撰写每个章节。以并行 fan-out 模式运行——通过 `Send()` 为每章触发一次调用。

**设计理由**：
- 使用 **Pro 模型**（从 Flash 切换而来），推理能力更强，章结构和字数约束遵从度更高。
- **逐章撰写，不是整篇。** 一次写一章，prompt 聚焦、输出可控。最后由 Assembler 拼接。
- **字数预算与知识点数量匹配**：beginner 2400字/章, intermediate 2800字/章, advanced 3200字/章。每章 3 个知识点，每个约 800-1000 字。
- **边界约束**：Writer prompt 注入章节位置（前一章/本章/后一章）和"禁止越界"清单（列出属于其他章的知识点），防止内容漂移。
- **修改模式**：Tier1 空章/超短章触发增量重试（只重写问题章节），字数问题由 Reviewer 处理。

**除错支持**：Writer 的 prompt 会保存到 `data/writer_prompt_debug.txt` 供线下排查。

### 2.5 审查（Agent 5）

**三级质量门**：

| 级别 | 名称 | 成本 | 检查内容 |
|------|------|------|----------|
| Tier1 | 机械检查 | 免费（纯 Python） | 空章、超短章(<50字)、代码块>30行 |
| Tier2 | 逐章审查 | 每章 1 次 Flash 调用 | 按清单逐项审查（5 个维度） |
| Tier3 | 结构审查 | 1 次 Flash 调用 | 跨章节结构、编排顺序、过渡 |

**Tier1 的设计意图**：只拦截明显坏的内容（整章为空或<50字标题壳子），不检查字数偏差（由 Reviewer 负责）。代码块>30行仅 minor 提示，不阻断。知识点覆盖不再在此检查——子串匹配假阳性太高，Reviewer 有语义理解。

**Tier2 并行执行**：通过 `Send("review_chapter", ...)` 并行审查所有章节。结果由 `assemble_reviews_node` 合并。

**核心原则——只修不改**：每次驳回时，`review_feedback.chapter_contents` 中包含原文。Writer 的修改模式只修复被标注的问题，保留已通过审查的内容。

**严重度判定**：只有知识点或代码完全缺失才标 critical。字数偏差、知识点偏短、术语缺解释一律标 minor。`assemble_reviews_node` 只在有 critical 时触发 Writer 重试。

**Tier2/3 使用 Flash 模型**：审查是判别任务（按清单逐项回答 通过/不通过），Flash 足够胜任，成本仅为 Pro 的 1/12。

### 2.6 组装器（元节点）

**职责**：将逐章草稿拼接为一篇完整博客。

处理部分重试场景：重试时只重写有问题的章节，其余从 `per_chapter_drafts` 按 `chapter_index` 复用。Assembler 将 carry-over 章节也写回 `per_chapter_drafts`，确保下游（Tier1/Reviewer）始终能看到完整章节。

---

## 3. 状态管理

### BlogGenState Schema

```python
class BlogGenState(TypedDict):
    messages: Annotated[list, add_messages]        # 聊天历史
    user_needs: dict                                # LearnerProfile
    knowledge_tree: dict                            # {domain, topics[]}
    chapter_plan: dict                              # {post_title, chapters[]}
    posts: list[dict]                               # 拆分计划
    current_post_index: int
    per_chapter_drafts: Annotated[list[dict], add]  # 并行撰写合并
    per_chapter_reviews: Annotated[list[dict], add] # 并行审查合并
    assembled_draft: str
    tier1_pass: bool
    draft: str
    writer_retry_count: int
    review_result: dict
    review_feedback: dict
    reject_level: str                               # "tier1"|"tier2"|"tier3"
    final: str
    completed_posts: list[dict]
    stage: str                                      # 流程控制
    # HITL 标记
    needs_approved: bool
    tree_approved: bool
    chapter_plan_approved: bool
    final_approved: bool
    session_created_at: str
```

### 关键状态模式

**并行合并器**：`per_chapter_drafts` 和 `per_chapter_reviews` 使用 `Annotated[list[dict], add]`。LangGraph 的 reducer 自动拼接并行 `Send()` 调用返回的结果列表。

**stage 作为路由信号**：`stage` 字段驱动所有路由决策。每个节点设置下一个 stage，`route_after_*` 函数读取它。没有隐式状态机。

**重试追踪**：`writer_retry_count` 在 `writer_batch_node` 中递增。`route_after_review` 检查是否超过 `MAX_REVIEW_RETRIES`（当前为 2）。达到上限后强制通过，不阻塞管线。

---

## 4. 检查点方案

当前使用 **SqliteSaver**，将 LangGraph 状态持久化到 SQLite 文件。

### 实现

```python
# src/graph/builder.py
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

def compile_graph(interrupt_after=None):
    conn = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    return build_graph().compile(
        checkpointer=checkpointer,
        interrupt_after=interrupt_after or [],
    )
```

`from_conn_string()` 返回 context manager，无法直接传给 `compile()`。因此直接创建 `sqlite3.connect()` 连接（`check_same_thread=False` 是 LangGraph 官方推荐配置），传给 `SqliteSaver(conn)`。

### 相比 MemorySaver 的优势

| | MemorySaver（旧） | SqliteSaver（当前） |
|---|---|---|
| 状态持久化 | ❌ 重启丢失 | ✅ 存到 SQLite |
| 崩溃恢复 | ❌ 重来 | ✅ 可继续 |
| 多会话隔离 | ❌ | ✅ |
| 新增依赖 | 无 | sqlite3（标准库）、langgraph-checkpoint-sqlite |

### 待做

- 旧会话清理策略（7 天未活跃自动删除），当前磁盘占用可忽略

---

## 5. 工具调用系统

### 设计

`_run_with_tools()` 实现了有界的工具调用循环：

1. 绑定工具到 LLM → SystemMessage + HumanMessage
2. 每轮（最多 `max_rounds` 轮）：
   - 检查剩余时间（<5 秒 → 跳出循环）
   - LLM 调用
   - 无 tool_call → 返回内容
   - 执行每个工具，追加 ToolMessage
   - 达到硬限制 → 强制输出最终结果
3. 最终兜底调用 → 返回内容

### 各 Agent 限制（config.py 配置）

| Agent | 最大轮次 | 最大工具调用次数 | 最大耗时 |
|-------|---------|---------------|---------|
| knowledge_tree | 2 | 1 | 180s |
| chapter_planner | 2 | 1 | 120s |
| writer_chapter | 1 | 1 | 300s |
| reviewer_chapter | 1 | 1 | 120s |

**设计原则——保守默认值。** 工具预算刻意设得很小——一次搜索通常足够。这避免了 Agent 陷入搜索死循环消耗大量 API 费用。

### 工具结果截断

超过 4000 字符的结果会被截断，防止上下文爆炸。4000 字符是一个经验值：足以容纳 Tavily 搜索结果（10 条的标题+摘要），又不会撑爆对话窗口。

---

## 6. RAG 系统

### 架构

```
Query → [BM25 (关键词)] + [向量 (语义)] → RRF 融合 → [Reranker (可选)] → Top-K
```

### 组件选择

- **BGE Embedding**（bge-large-zh-v1.5）：选型时中文文本 embedding 效果最好的开源模型之一。
- **ChromaDB**：轻量级嵌入式向量库，不需要外部服务。
- **BM25**（rank-bm25）：关键词匹配能捕获中文技术术语（embedding 有时会遗漏）。
- **RRF（Reciprocal Rank Fusion）**：简单的无参数融合算法，不需要调参。
- **BGE Reranker**（bge-reranker-v2-m3）：可选的第二遍精排。Cross-encoder 对 Top 结果重新打分。

### 优雅降级

所有 RAG 组件使用惰性加载+回退。如果 `chromadb` 或 `sentence_transformers` 未安装：
- `_rag_available` → False
- `seed_from_markdown()` 和 `query_vector_store()` 变成空操作

这让 Agent 管线在无 RAG 时也能正常运行——适合开发和 CI 场景。

---

## 7. 监控

### 数据采集

`BlogGenMonitorCallback`（LangChain `BaseCallbackHandler`）挂在：
- `on_llm_start` / `on_llm_end` → Token 用量、模型名、延迟
- `on_chain_start` / `on_chain_end` → 每个节点的耗时

### 输出

- **JSONL 日志**（`data/logs.jsonl`）：每次图谱调用一条记录，含逐节点分解
- **内存缓冲区**：`_session_logs` 滚动列表，供 Streamlit 侧边栏显示
- **Token 总计**：按会话追踪，显示在 UI 中

---

## 8. 设计权衡与已知局限

### 已接受的权衡

1. **Writer 用 Flash 模型。** MVP 阶段优先控制成本，生成质量后续实测后再决定是否切 Pro。
2. **无流式输出。** Streamlit 的 `st.spinner()` + 阻塞式 invoke 比流式 SSE 简单。代价是 Writer/Reviewer 长时间调用时用户感知延迟高。
3. **逐章独立审查。** Tier3 的结构审查会检查跨章问题，但相邻章节间的细粒度内容衔接问题可能漏过。可接受，因为章节设计本身就是相对独立的单元。

### 已知局限

1. **Tier1 子串匹配粗糙。** 能拦截明显缺失，但无法判断语义完整性。Tier2 的 LLM 审查补足这一层。实际拦截效果后续需要评估。
2. **单一 LLM 提供商。** 系统与 DeepSeek API 耦合。换其他提供商需要改 base_url、模型名，以及重新测试所有 Agent 的输出格式兼容性。
3. **Prompt 无 A/B 测试。** 修改 prompt 后没有和基线对比的机制，存在回归风险。

---

## 9. 测试策略

### 测试类别

| 类别 | 数量 | 覆盖内容 |
|------|------|---------|
| Schema 校验 | 36 | Pydantic 模型、边界值、模糊输入规范化 |
| 路由逻辑 | 20 | 每个条件边、重试上限、fan-out 决策 |
| 节点逻辑 | ~40 | Mock LLM 调用、Agent 行为正确性 |
| 工具系统 | 13 | 工具调用、限制、截断、错误处理 |
| 集成测试 | ~15 | 多节点流程（使用模拟 LLM） |
| E2E | ~12 | 从输入到输出的完整管线 |
| 回归测试 | ~10 | 已修复 Bug 编码为测试 |
| 解析鲁棒性 | ~10 | JSON/Markdown 解析边界情况 |

### 测试理念

- **不调用真实 LLM。** 所有测试用 `unittest.mock` 替换 `get_llm()`/`get_fast_llm()`。这样保持测试快速（300 个测试约 3 秒）且免费。
- **集成测试用录制数据。** `test_integration.py` 用预录的 LLM 输出测试多 Agent 流程。
- **可选依赖用 skip 标记。** `@pytest.mark.skipif(not HAS_LANGGRAPH, ...)` 确保在不完整环境上也能跑。

---

## 10. 配置设计

### 两级配置

**`.env` 管密钥**（API key、功能开关）：
- `DEEPSEEK_API_KEY` — 导入时校验，拒绝占位符值
- `TAVILY_API_KEY` — 仅当 `ENABLE_NETWORK_SEARCH=true` 时需要
- `ENABLE_NETWORK_SEARCH` — 成本控制开关

**`config.py` 管行为**（深度规则、工具限制、风格规则）：
- `DEPTH_RULES` — 按水平（beginner/intermediate/advanced）配置字数预算、代码要求
- `STYLE_RULES` — 写作风格（practical/theoretical/balanced）
- `MAX_TOOL_*` — 每个 Agent 的工具调用限制（次数、轮次、超时）
- `MAX_REVIEW_RETRIES` — 全局重试上限（当前 2）

### 设计原则：默认保守

所有工具限制刻意设小。管线应优先在不使用外部工具的情况下完成（依靠 LLM 自身知识）。Web 搜索通过 `ENABLE_NETWORK_SEARCH` 开关控制。这保持了成本可预测。
