# BlogGen — 多 Agent 协作博客生成系统

BlogGen 是一个基于 LangGraph 的多 Agent 协作博客生成系统。5 个 AI Agent 分工协作——需求对齐 → 知识树构建 → 章节规划 → 撰写 → 审查——通过人在回路（HITL）机制让用户把控关键决策。

## 工作流

```
用户输入话题 → 需求对齐 → 知识树构建 → 章节规划 → 撰写(并行) → 审查(并行)
                HITL ✓      HITL ✓       HITL ✓                  HITL ✓
                              ↓
                        最终博客输出
```

## 5 个 Agent

| Agent | 职责 | 输出 |
|-------|------|------|
| 需求对齐 | 对话式需求收集，分析读者水平、学习目标、风格偏好 | LearnerProfile |
| 知识树构建 | 研究主题，构建有序知识树，可选联网搜索 | KnowledgeTree |
| 章节规划 | 将知识点归组为章节，按预算拆分为单篇或多篇博客 | ChapterPlan |
| 撰写 | 并行撰写各章节，注入代码示例和生活化类比 | Markdown 草稿 |
| 审查 | 三级质量门（代码检查 → 逐章审查 → 结构审查），驳回或通过 | ReviewResult |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

在项目根目录创建 `.env` 文件：

```env
DEEPSEEK_API_KEY=sk-你的key
# 可选：联网搜索
ENABLE_NETWORK_SEARCH=false
TAVILY_API_KEY=tvly-你的key
```

### 3. 启动

```bash
streamlit run main.py
```

浏览器打开 `http://localhost:8501`，在聊天界面输入你想写的博客话题，按提示逐步确认即可。

## 功能特性

- **人在回路**：4 个审批节点（需求确认、知识树审批、章节规划审批、审查结果决策）
- **并行撰写**：所有章节通过 LangGraph `Send()` 并行生成，而非逐章串行
- **三级质量门**：Tier1 免费拦截空章/超短章 → Tier2 逐章审查 → Tier3 结构审查，minor 不阻断
- **重试优化**：只重试有问题的章节（增量），结构审查结果复用
- **RAG 增强**：ChromaDB 向量库 + BGE 中文 Embedding + BM25 混合检索
- **读者分层**：按水平调整字数+章节数（初学 2400字/章 ≤5章, 中级 2800字/章 ≤6章, 进阶 3200字/章 ≤7章）
- **单概念知识树**：每个知识点必须是单一概念（禁止"与""及"合并），粒度均匀
- **监控面板**：实时 Token 用量、节点耗时、日志查看
- **离线优先**：RAG 和 Web 搜索均可降级，核心管线不依赖外部服务

## 项目结构

```
BlogGen/
├── main.py              # Streamlit 入口（UI + HITL 路由）
├── src/
│   ├── agents/
│   │   ├── nodes.py     # 7 个 Agent 节点实现（~1320 行）
│   │   └── prompts.py   # 各 Agent 的系统提示词
│   ├── graph/
│   │   ├── builder.py   # LangGraph 图构建 + 条件路由
│   │   ├── state.py     # BlogGenState 类型定义
│   │   └── session.py   # Streamlit ↔ LangGraph 桥接
│   ├── config.py        # LLM 配置、深度/风格规则、工具限制
│   ├── schemas.py       # Pydantic 数据契约（含模糊输入规范化）
│   ├── llm_utils.py     # LLM 调用工具（JSON 提取、重试、工具调用循环）
│   ├── monitor.py       # LangChain 回调：Token 追踪 + JSONL 日志
│   ├── tools/           # tavily_search、query_vector_store、fetch_page
│   ├── rag/             # ChromaDB + BGE Embedding + BM25（可降级）
│   └── ui/components.py # Streamlit UI 组件
├── docs/
│   └── design.md        # 设计文档：架构决策、Agent 设计、权衡分析
├── tests/               # 300 个测试（全部通过，约 3 秒）
├── data/                # ChromaDB + SQLite 检查点(持久化) + 日志（不入 git）
└── outputs/             # 生成的博客（不入 git）
```

## 技术栈

- **编排引擎**：LangGraph（StateGraph + conditional edges + Send fan-out）
- **LLM**：DeepSeek V4 Pro（推理）/ Flash（生成），通过 ChatOpenAI 兼容接口
- **UI**：Streamlit（wide 布局，callback 模式）
- **RAG**：ChromaDB + BAAI/bge-large-zh-v1.5 + BM25 + RRF 融合
- **数据校验**：Pydantic v2（field_validator + model_validator）
- **监控**：LangChain BaseCallbackHandler → JSONL 日志
- **状态管理**：SqliteSaver 持久化到 SQLite（Streamlit 重启不丢进度）

## 运行测试

```bash
# 全部测试（300 个，约 3 秒）
pytest tests/ -v

# 跳过集成测试
pytest tests/ -m "not integration" -v

# 单个测试文件
pytest tests/test_schemas.py -v
```

## 设计文档

详见 [docs/design.md](docs/design.md)，涵盖：
- 为什么选 LangGraph / DeepSeek / Streamlit / Pydantic
- 5 个 Agent 各自的设计理由和边界处理
- 状态管理、工具调用系统、RAG 架构
- 三级质量门的设计意图
- 已知权衡和局限性

## 安装方式

```bash
git clone <repo-url>
cd BlogGen
pip install -r requirements.txt
cp .env.example .env  # 编辑填入 API Key
streamlit run main.py
```

## License

MIT
