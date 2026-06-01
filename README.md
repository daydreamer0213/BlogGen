# BlogGen

[![Test](https://github.com/daydreamer0213/BlogGen/actions/workflows/test.yml/badge.svg)](https://github.com/daydreamer0213/BlogGen/actions/workflows/test.yml)
[![Python](https://img.shields.io/badge/python-3.11+-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

5 个 AI Agent 协作，把一句话话题变成一篇结构严谨、有代码示例的技术博客。

```
输入: "我想学RAG，我是初学者"  →  输出: 12000字的技术博文，含代码和类比
```

## 快速开始

```bash
git clone https://github.com/daydreamer0213/BlogGen.git
cd BlogGen
pip install -r requirements.txt
cp .env.example .env   # 编辑，填入 DEEPSEEK_API_KEY
streamlit run main.py
```

浏览器打开 `http://localhost:8501`，输入你想写的话题，按提示确认每一步。

## 管线

```
用户输入 → 需求对齐 → 知识树构建 → 章节规划 → Writer×N (并行) → Reviewer×N (并行)
             HITL ✓      HITL ✓       HITL ✓                       ↓
                                                               accept / retry
```

## 5 个 Agent

| Agent | 模型 | 职责 |
|-------|------|------|
| 需求对齐 | Pro | 多轮对话收集读者水平、目标、风格偏好 |
| 知识树构建 | Flash | 研究主题，按学习顺序列出单一概念知识点 |
| 章节规划 | Flash | 将知识点归组为章节（≤5-7章），每章3个点 |
| 撰写 | Pro ×N并行 | 逐章撰写：类比 → 原理 → 代码示例 → 拓展思考 |
| 审查 | Flash ×N并行 | 三级质量门：空章拦截 → 清单审查 → 结构检查 |

## 功能

- **人在回路** — 4 个审批节点，用户可以确认/修改中间产物
- **并行撰写** — LangGraph `Send()` fan-out，6 章同时写而非逐章串行
- **三级审查** — Tier1 免费拦截空章超短章；Tier2/3 清单+结构审查；minor 不阻断
- **增量重试** — 只重写有问题的章节，复用审查结果
- **读者分层** — 初学者 2400字/章×5章，中级 2800字/章×6章，进阶 3200字/章×7章
- **成本可控** — 单篇 ~$0.20

## 命令

```bash
# 安装
pip install -r requirements.txt

# 开发测试
pip install -r requirements-ci.txt    # 带上 pytest
pytest tests/ -v                      # 336 tests, ~2s

# 跳过集成测试
pytest tests/ -m "not integration" -v

# 无头管线测试（全自动，不启动 UI）
python run_headless.py --timeout 600

# 启动
streamlit run main.py
```

## 项目结构

```
BlogGen/
├── main.py              # Streamlit 入口
├── run_headless.py      # 无头测试脚本
├── src/
│   ├── agents/
│   │   ├── nodes.py     # 7 个 Agent 节点
│   │   └── prompts.py   # 系统提示词
│   ├── graph/
│   │   ├── builder.py   # LangGraph 图 + 路由
│   │   ├── state.py     # BlogGenState
│   │   └── session.py   # Streamlit ↔ LangGraph
│   ├── config.py        # 深度规则、工具限制
│   ├── schemas.py       # Pydantic 数据契约
│   ├── llm_utils.py     # LLM 调用、JSON 提取
│   ├── monitor.py       # Token 追踪 + JSONL 日志
│   ├── tools/           # Web 搜索、向量检索
│   ├── rag/             # ChromaDB + BGE + BM25
│   └── ui/              # Streamlit 组件
├── tests/               # 336 tests, ~2s
├── docs/design.md       # 设计文档
├── data/                # ChromaDB + SQLite (gitignored)
└── outputs/             # 生成博客 (gitignored)
```

## 设计

详见 [docs/design.md](docs/design.md)。

## License

MIT
