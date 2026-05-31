# BlogGen — Multi-Agent Collaborative Blog Generation

BlogGen 是一个基于 LangGraph 的多 Agent 协作博客生成系统。5 个 AI Agent 分工协作：需求对齐 → 知识树构建 → 章节规划 → 撰写 → 审查，通过人在回路（HITL）机制保证质量。

## 工作流

```
用户输入话题 → [NeedsAlignment] → 学情分析 → HITL ✓
                              ↓
                    [KnowledgeTree] → 知识树 → HITL ✓
                              ↓
                    [ChapterPlanner] → 章节规划 → HITL ✓
                              ↓
                    [Writer] → 逐章撰写（并行 fan-out）
                              ↓
                    [Reviewer] → 逐章审查 + 结构审查 → 通过/驳回
                              ↓
                    最终博客输出
```

## 5 个 Agent

| Agent | 职责 | 输出 |
|-------|------|------|
| NeedsAlignment | 对话式需求收集，分析读者水平、目标、风格偏好 | LearnerProfile |
| KnowledgeTreeBuilder | 研究主题，构建有序知识树，必要时联网搜索 | KnowledgeTree |
| ChapterPlanner | 将知识点分组为章节，保证逻辑递进 | ChapterPlan |
| Writer | 并行撰写各章节，注入代码示例和生活化类比 | Markdown 草稿 |
| Reviewer | 逐章 checklist 审查 + 跨章节结构检查，驳回/接受 | ReviewResult |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

在项目根目录创建 `.env` 文件：

```env
DEEPSEEK_API_KEY=sk-your-key-here
# 可选：联网搜索
ENABLE_NETWORK_SEARCH=false
TAVILY_API_KEY=tvly-your-key-here
```

### 3. 启动

```bash
streamlit run main.py
```

浏览器打开 `http://localhost:8501`，在聊天界面输入你想写的博客话题。

## 功能特性

- **人在回路（HITL）**：需求确认、知识树审批、章节规划审批、审查结果决策
- **并行撰写**：所有章节通过 LangGraph `Send()` 并行生成
- **质量门**：Tier1（自动检查）+ Tier2（逐章审查）+ Tier3（结构审查），最多重试 2 次
- **多篇博客**：支持从知识树自动拆分为多篇博客，依次生成
- **RAG 增强**：ChromaDB 向量库 + BGE 中文 Embedding + BM25 混合检索
- **读者分层**：根据初学者/中级/进阶自动调整深度、字数、代码比例
- **监控面板**：实时 Token 用量、节点耗时、日志查看

## 项目结构

```
BlogGen/
├── main.py              # Streamlit 入口（UI + HITL 路由）
├── src/
│   ├── agents/
│   │   ├── nodes.py     # 7 个 Agent 节点实现
│   │   └── prompts.py   # 各 Agent 的系统提示词
│   ├── graph/
│   │   ├── builder.py   # LangGraph 图构建 + 路由逻辑
│   │   ├── state.py     # BlogGenState 类型定义
│   │   └── session.py   # Streamlit ↔ LangGraph 桥接
│   ├── config.py        # LLM 配置、深度规则、工具限制
│   ├── schemas.py       # Pydantic 数据契约（含模糊输入规范化）
│   ├── llm_utils.py     # LLM 调用工具（JSON 提取、重试）
│   ├── monitor.py       # LangChain 回调：Token 追踪 + JSONL 日志
│   ├── tools/           # tavily_search, query_vector_store, fetch_page
│   ├── rag/             # ChromaDB + BGE Embedding + BM25
│   └── ui/components.py # Streamlit UI 组件
├── tests/               # 300 个测试（全部通过，~3s）
│   ├── conftest.py      # 共享 fixtures + 可选依赖跳过标记
│   ├── test_schemas.py  # Pydantic 模型边界测试（36 个）
│   ├── test_nodes*.py   # Agent 节点测试（~1200 行）
│   ├── test_routing.py  # 图路由逻辑测试（20 个）
│   ├── test_integration.py # 集成测试
│   └── test_e2e.py      # 端到端测试
├── data/                # ChromaDB 向量库 + SQLite 检查点 + 日志
└── outputs/             # 生成的博客（Markdown + JSON 摘要）
```

## 技术栈

- **编排**: LangGraph（StateGraph + conditional edges + Send fan-out）
- **LLM**: DeepSeek V4 Pro（主模型）/ Flash（快速模型）
- **UI**: Streamlit（wide 布局）
- **RAG**: ChromaDB + BAAI/bge-large-zh-v1.5 + BM25
- **数据校验**: Pydantic v2（field_validator + model_validator）
- **监控**: LangChain BaseCallbackHandler → JSONL 日志

## 运行测试

```bash
# 全部测试
pytest tests/ -v

# 跳过慢速/集成测试
pytest tests/ -m "not slow and not integration" -v

# 单个测试文件
pytest tests/test_schemas.py -v
```

## License

MIT
