# BlogGen Design Document

## System Overview

BlogGen is a multi-agent collaborative blog generation system that transforms a user's learning topic into a polished, pedagogically sound technical blog post. It orchestrates 5 AI agents through a LangGraph state machine, with human-in-the-loop (HITL) checkpoints at key decision points.

### Pipeline

```
User Input → NeedsAlignment → KnowledgeTree → ChapterPlanner → Writer(fan-out) → Reviewer(fan-out)
                 HITL ✓          HITL ✓         HITL ✓                              HITL ✓
```

At each HITL checkpoint, the Streamlit UI renders the agent's output and waits for user approval before advancing to the next stage.

---

## 1. Architecture Decisions

### 1.1 Why LangGraph over custom orchestration

- **Checkpointing built-in.** MemorySaver gives us pause/resume for HITL without implementing our own state serialization.
- **Fan-out via Send().** LangGraph 1.x `Send()` from conditional edges gives parallel chapter execution with reducer-based merge — no manual thread pools.
- **Conditional routing.** `add_conditional_edges()` lets each node decide the next step based on state, keeping routing logic explicit.
- **Explicit state schema.** BlogGenState TypedDict with `Annotated[list, add]` reducers makes state mutations traceable.

Tradeoff: LangGraph adds a dependency, but the alternative (hand-rolled state machine + threading) would be significantly more code for the same correctness guarantees.

### 1.2 Why DeepSeek as the LLM backend

- **Chinese language quality.** BGE embeddings (bge-large-zh-v1.5) and DeepSeek's native Chinese support are essential for the target audience.
- **Two-tier model strategy.** DeepSeek V4 Pro for complex reasoning (NeedsAlignment, Reviewer), Flash for generation-heavy tasks (Writer). Flash is ~12x cheaper with only 1.9 point quality gap on long-form writing.
- **OpenAI-compatible API.** ChatOpenAI client works without vendor-specific SDK.

### 1.3 Why Streamlit for UI

- **Python-native.** No frontend build step. All agents are Python, so keeping the UI in Python avoids a language boundary.
- **Session state model.** `st.session_state` maps cleanly to BlogGenSession lifecycle — one session per browser tab.
- **HITL UX.** Streamlit's callback pattern (`on_click`) + `st.rerun()` gives us a responsive approval flow without WebSocket complexity.

### 1.4 Why Pydantic for data contracts

- **Validation at boundaries.** Every inter-agent handoff (LearnerProfile, KnowledgeTree, ChapterPlan, ReviewResult) goes through `validate_or_raise()`.
- **Fuzzy input normalization.** `normalize_level_str()` handles Chinese proficiency descriptors ("小白"→beginner, "精通"→advanced) so the LLM doesn't need to get the exact string right.
- **Schema drift detection.** If an agent's output format changes, Pydantic raises immediately rather than propagating bad data downstream.

---

## 2. Agent Design

### 2.1 NeedsAlignment (Agent 1)

**Responsibility**: Conversational requirements gathering. Extracts structured LearnerProfile from multi-turn chat.

**Design rationale**:
- The `needs_alignment` stage loops — the graph routes back to the same node until all required fields (domain, level, goal) are present.
- This is a HITL checkpoint by design: the node interrupts after each LLM response so the user can continue chatting or approve.
- `extract_json()` is called on every LLM response; missing fields trigger a follow-up question.

**Edge case handling**:
- Empty history → stays in needs_alignment (UI shows greeting)
- Partial profile → LLM asks targeted follow-up question (1-2 missing fields only)
- Fuzzy level descriptors → normalized by `normalize_level_str()`

### 2.2 KnowledgeTreeBuilder (Agent 2)

**Responsibility**: Research the domain and produce an ordered list of knowledge points (topics) from beginner to mastery.

**Design rationale**:
- Uses the **fast model** (Flash) — knowledge tree is structural, not creative. Speed matters more than nuance.
- Has access to web search (`tavily_search`) and local RAG (`query_vector_store`) for research.
- **2026 pattern**: Instead of unlimited web search, agents have a small tool call budget (typically 1 call). One search with `max_results=10` is usually sufficient.
- Output format is deliberately simple: `# Domain\n- Topic 1\n- Topic 2`. Simple format = fewer parsing failures.

**Retry logic**:
- First attempt: standard tool-calling flow
- Validation failure → retry with explicit format instructions
- Retry failure → accept whatever topics were parsed (graceful degradation)

### 2.3 ChapterPlanner (Agent 3)

**Responsibility**: Group knowledge points into chapters, then split chapters into blog posts by word budget.

**Key design decision — planner doesn't design narrative**:
The ChapterPlanner only groups topics. Writer owns narrative structure (core questions, analogies, code examples). This separation prevents the planner from making creative decisions that constrain the writer.

**Split algorithm** (`split_chapters_by_budget`):
- Recursive binary split at chapter boundaries
- Each post gets `max_words_per_chapter × 5` word budget
- Merge trailing fragments <30% of budget into the previous post
- Uses chapter boundaries as split points (natural semantic breaks)

### 2.4 Writer (Agent 4)

**Responsibility**: Write each chapter as standalone content. Operates in fan-out mode — one invocation per chapter via `Send()`.

**Design rationale**:
- **Flash model** for generation — DeepSeek's own evaluation shows Flash is only 1.9 points behind Pro on long-form Chinese writing, at ~12x lower cost.
- **Per-chapter, not per-post.** Writing one chapter at a time keeps prompts focused and output manageable. The Assembler concatenates.
- **Fix modes** depend on which review tier rejected:
  - Tier1 (code-level): Add missing topics, trim code blocks, cut words. Preserve other content.
  - Tier2 (structure-level): Reorder paragraphs, improve transitions. Preserve paragraph content.
  - Tier3 (content-level): Modify only flagged paragraphs. Everything else preserved verbatim.
- **Word budget** is repeated at the start and end of each prompt — the "primacy/recency" effect ensures it's followed.

**Debug support**: Writer prompts are saved to `data/writer_prompt_debug.txt` for offline inspection.

### 2.5 Reviewer (Agent 5)

**Three-tier quality gate**:

| Tier | Name | Cost | What it checks |
|------|------|------|----------------|
| Tier1 | Code-level | Free (pure Python) | Topic coverage, word count, code block size |
| Tier2 | Per-chapter | 1 LLM call/chapter | Checklist review per chapter (5 dimensions) |
| Tier3 | Structure | 1 LLM call | Cross-chapter structure, ordering, transitions |

**Tier1 is free by design.** Topic coverage uses substring matching (`core_term in chapter_content`), word count is `len(content)`, code block lines counted via regex. These checks catch 80% of issues without burning API credits.

**Tier2 fan-out**: Each chapter reviewed in parallel via `Send("review_chapter", ...)`. Reviews merged by `assemble_reviews_node`.

**Design principle — fix, don't rewrite**: Each rejection includes the original chapter content in `review_feedback.chapter_contents`. The Writer's fix mode modifies only the flagged issues, preserving the rest. This prevents review-fix oscillation (Writer changing good content while fixing bad).

### 2.6 Assembler (meta-node)

**Responsibility**: Concatenate per-chapter drafts into a single blog post.

Handles partial retry: if some chapters weren't re-written (no issues), `_extract_chapter_draft()` pulls them from the old assembled draft via regex heading matching.

---

## 3. State Management

### BlogGenState Schema

```python
class BlogGenState(TypedDict):
    messages: Annotated[list, add_messages]        # Chat history (NeedsAlignment)
    user_needs: dict                                # LearnerProfile
    knowledge_tree: dict                            # {domain, topics[]}
    chapter_plan: dict                              # {post_title, chapters[]}
    posts: list[dict]                               # Split plan [{title, chapter_indices}]
    current_post_index: int
    per_chapter_drafts: Annotated[list[dict], add]  # Fan-out accumulator
    per_chapter_reviews: Annotated[list[dict], add] # Fan-out accumulator
    assembled_draft: str
    tier1_pass: bool
    draft: str
    writer_retry_count: int
    review_result: dict
    review_feedback: dict
    reject_level: str                               # "tier1"|"tier2"|"tier3"
    final: str
    completed_posts: list[dict]
    stage: str                                      # Flow control
    # HITL flags
    needs_approved: bool
    tree_approved: bool
    chapter_plan_approved: bool
    final_approved: bool
    session_created_at: str
```

### Key state patterns

**Fan-out accumulators**: `per_chapter_drafts` and `per_chapter_reviews` use `Annotated[list[dict], add]`. LangGraph's reducer concatenates results from parallel `Send()` invocations.

**Stage as routing signal**: The `stage` field drives all routing decisions. Each node sets the next stage; `route_after_*` functions read it. No hidden state machines.

**Writer retry tracking**: `writer_retry_count` increments in `writer_batch_node`. `route_after_review` checks it against `MAX_REVIEW_RETRIES` (2). After 2 retries, the system forces acceptance.

---

## 4. Tool Calling System

### Design

`_run_with_tools()` in `nodes.py` implements a bounded tool-calling loop:

1. Bind tools to LLM → SystemMessage + HumanMessage
2. For each round (up to `max_rounds`):
   - Check remaining time (<5s → break)
   - LLM call
   - If no tool_calls → return content
   - Execute each tool, append ToolMessage
   - If hard limit reached → force final output
3. Final fallback call → return content

### Per-agent limits (from config)

| Agent | Max Rounds | Max Tool Calls | Max Time |
|-------|-----------|---------------|----------|
| knowledge_tree | 2 | 1 | 180s |
| chapter_planner | 2 | 1 | 120s |
| writer_chapter | 1 | 1 | 300s |
| reviewer_chapter | 1 | 1 | 120s |

**Conservative by design.** The tool budget is deliberately small — one web search is usually enough. This prevents agents from falling into infinite search loops.

### Tool result truncation

Results >4000 characters are truncated to prevent context explosion. The 4000-char limit was chosen empirically: it's enough for a Tavily search result (title + snippet for 10 results) without bloating the conversation.

---

## 5. RAG System

### Architecture

```
Query → [BM25 (keyword)] + [Vector (semantic)] → RRF merge → [Reranker (optional)] → Top-K
```

### Component choices

- **BGE Embeddings** (bge-large-zh-v1.5): Best-in-class Chinese text embeddings at the time of selection.
- **ChromaDB**: Lightweight, embedded vector store. No external service needed.
- **BM25 via rank-bm25**: Keyword matching catches Chinese technical terms that embeddings sometimes miss.
- **RRF (Reciprocal Rank Fusion)**: Simple, parameter-free fusion algorithm. No tuning needed.
- **BGE Reranker** (bge-reranker-v2-m3): Optional second pass. Cross-encoder re-ranks top results for precision.

### Graceful degradation

All RAG components use lazy init with fallback. If `chromadb` or `sentence_transformers` isn't installed:
- `_rag_available` → False
- `seed_from_markdown()` and `query_vector_store()` become no-ops

This lets the agent pipeline work without RAG — useful for development and CI.

---

## 6. Monitoring

### Data collection

`BlogGenMonitorCallback` (LangChain `BaseCallbackHandler`) hooks into:
- `on_llm_start` / `on_llm_end` → token counts, model name, latency
- `on_chain_start` / `on_chain_end` → per-node timing

### Output

- **JSONL logs** (`data/logs.jsonl`): One entry per graph invocation, with per-node breakdown
- **In-memory buffer**: `_session_logs` rolling list for the Streamlit sidebar
- **Token totals**: Tracked per-session, displayed in UI

---

## 7. Design Tradeoffs & Known Limitations

### Accepted tradeoffs

1. **MemorySaver, not SQLite checkpointer.** State is lost on page refresh. Acceptable for a Streamlit app where session lifetime = browser tab lifetime. The `_seed_knowledge_base()` function re-seeds on each new session.

2. **Writer uses Flash, not Pro.** The quality gap (1.9 points) is acceptable for long-form generation. If future evaluations show degradation, switching to Pro is a one-line config change.

3. **No streaming output.** Streamlit's `st.spinner()` + blocking invoke is simpler than streaming SSE chunks. The tradeoff is higher perceived latency for long Writer/Reviewer calls.

4. **Per-chapter review, not holistic.** The structure reviewer catches cross-chapter issues at Tier3, but content-level flow problems between chapters may be missed. Acceptable because chapters are designed to be self-contained units.

### Known limitations

1. **No incremental state persistence.** If the Streamlit server restarts, all in-progress sessions are lost. A SQLite-backed checkpointer would fix this.
2. **Tier1 coverage check is substring-based.** It can miss semantic omissions (topic mentioned but not actually explained). Tier2 fills this gap with LLM review.
3. **Single LLM provider.** The system is coupled to DeepSeek. Switching to another provider requires changing the ChatOpenAI base_url and model names.
4. **No A/B testing for prompts.** Prompt changes are not evaluated against a baseline. Regression risk on prompt edits.

---

## 8. Testing Strategy

### Test categories

| Category | Count | What it covers |
|----------|-------|----------------|
| Schema validation | 36 | Pydantic models, boundary values, fuzzy normalization |
| Routing logic | 20 | Every conditional edge, retry limits, fan-out decisions |
| Node logic | ~40 | Mocked LLM calls, agent behavior correctness |
| Tool system | 13 | Tool invocation, limits, truncation, error handling |
| Integration | ~15 | Multi-node flows with real (mocked) LLM |
| E2E | ~12 | Full pipeline from input to output |
| Regression | ~10 | Fixed bugs encoded as tests |
| Parser robustness | ~10 | JSON/Markdown parsing edge cases |

### Testing philosophy

- **No real LLM calls.** All tests mock `get_llm()` / `get_fast_llm()` via `unittest.mock`. This keeps the suite fast (~2.75s for 300 tests) and free.
- **Integration tests use recorded responses.** The `test_integration.py` file tests multi-agent flows with pre-recorded outputs.
- **Skip markers for optional deps.** `@pytest.mark.skipif(not HAS_LANGGRAPH, ...)` ensures tests still run on machines without the full dependency stack.

---

## 9. Configuration Design

### Two-tier config

**`.env` for secrets** (API keys, feature flags):
- `DEEPSEEK_API_KEY` — validated at import time, rejects placeholder values
- `TAVILY_API_KEY` — only required if `ENABLE_NETWORK_SEARCH=true`
- `ENABLE_NETWORK_SEARCH` — feature flag for cost control

**`config.py` for behavior** (depth rules, tool limits, style rules):
- `DEPTH_RULES` — per-level (beginner/intermediate/advanced) word budgets, code requirements
- `STYLE_RULES` — writing style preferences (practical/theoretical/balanced)
- `MAX_TOOL_*` — per-agent tool calling limits (calls, rounds, timeout)
- `MAX_REVIEW_RETRIES` — global retry cap (currently 2)

### Design principle: conservative defaults

All tool limits are deliberately small. The pipeline should complete without tools (offline knowledge) whenever possible. Web search is opt-in via `ENABLE_NETWORK_SEARCH`. This keeps costs predictable.
