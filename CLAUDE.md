# BlogGen — Multi-Agent Collaborative Blog Generation System

5-agent LangGraph pipeline: NeedsAlignment → KnowledgeTree → ChapterPlanner → Writer → Reviewer.
Built with Streamlit UI + LangChain/LangGraph + ChromaDB RAG.

## Project structure

```
BlogGen/
├── main.py              # Streamlit entry point (UI + HITL routing)
├── src/
│   ├── agents/nodes.py  # 7 agent node implementations (~1320 lines)
│   ├── agents/prompts.py# System prompts for each agent
│   ├── graph/builder.py # LangGraph StateGraph + routing (fan-out via Send)
│   ├── graph/state.py   # BlogGenState TypedDict + initial_state()
│   ├── graph/session.py # Streamlit ↔ LangGraph bridge (BlogGenSession)
│   ├── config.py        # LLM config, depth/style rules, tool limits
│   ├── schemas.py       # Pydantic contracts (LearnerProfile, KnowledgeTree, etc.)
│   ├── llm_utils.py     # get_llm, safe_extract_json, invoke_with_retry
│   ├── monitor.py       # LangChain callback: token tracking + JSONL logging
│   ├── tools/           # tavily_search, query_vector_store, fetch_page
│   ├── rag/             # ChromaDB vector store with BGE embeddings + BM25
│   └── ui/components.py # Streamlit UI components
├── tests/               # 300 tests (schemas, routing, nodes, tools, e2e, regression)
├── docs/
│   └── design.md        # Design doc: architecture decisions, agent pipeline, tradeoffs
├── data/                # ChromaDB + SQLite checkpoints (持久化) + logs
└── outputs/             # Generated blog posts (markdown + summary JSON)
```

## Commands

```bash
# Install
pip install -r requirements.txt

# Run the app
streamlit run main.py

# Run tests (300 tests, ~3s)
pytest tests/ -v

# Run fast tests only (skip integration tests)
pytest tests/ -m "not integration" -v
```

## Key conventions

- **State persistence:** SqliteSaver 持久化到 data/checkpoints.db（Streamlit 重启不丢进度）
- **State management:** LangGraph TypedDict with `add` reducer for fan-out lists (per_chapter_drafts, per_chapter_reviews)
- **Fan-out pattern:** `Send("write_chapter", ...)` and `Send("review_chapter", ...)` from conditional edges
- **HITL checkpoints:** Graph interrupts after needs_alignment, knowledge_tree, chapter_planner, review_batch
- **Agent naming convention:** `_node` suffix for all graph nodes, matching keys in add_node()
- **Tool calling:** `_run_with_tools()` in nodes.py with per-agent limits (max calls, rounds, timeout)
- **JSON extraction:** LLM outputs JSON in markdown fences → `safe_extract_json()` with regex fallback
- **Config:** All env vars validated at import time via `_require()`, placeholder detection
- **Word budget:** beginner=1800字/chapter, intermediate=2000字/chapter. 每章聚焦2-3个核心知识点
- **Tier1 retry:** 纯字数驳回不传原文（防 prompt 膨胀），内容问题才带原文定位

## Workflow preferences (superpowers)

- Use **worktrees** for feature work (`using-git-worktrees` skill)
- Use **TDD** for all implementation (`test-driven-development` skill)
- Use **systematic-debugging** before proposing fixes for bugs
- Use **writing-plans** for multi-step tasks
- Tests must be passing (`pytest tests/`) before claiming work is complete
- **每次修改后更新 CLAUDE.md / README.md / docs/design.md**，保持文档与代码同步

---

# Behavioral Guidelines

Guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
