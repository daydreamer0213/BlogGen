"""Agent node implementations for the 5-agent BlogGen pipeline.

Each node receives state → returns a partial state update dict.
Agents with tools use function calling via _run_with_tools().

Node list:
  1. needs_alignment_node    — chat-based info collection → LearnerProfile
  2. knowledge_tree_node     — research + ordered topic list → KnowledgeTree
  3. chapter_planner_node    — group topics into chapters → ChapterPlan
  4. writer_single_chapter_node / writer_batch_node → per-chapter Markdown
  5. assembler_node          — concatenate chapters into final post
  6. reviewer_single_chapter_node / structure_reviewer_node → checklist review
  7. assemble_reviews_node   — merge per-chapter + structure reviews → accept/reject

All nodes accept a dict state and return a dict partial update.
LangGraph merges partial updates via the StateGraph reducer.
"""
import json
import time
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage

from src.config import (
    DEPTH_RULES, MAX_REVIEW_RETRIES,
    MAX_TOOL_CALLS_PER_AGENT, MAX_TOOL_SEC_PER_AGENT, MAX_TOOL_ROUNDS_PER_AGENT,
)
from src.llm_utils import (
    get_llm, get_fast_llm,
    safe_extract_json,
    invoke_with_retry,
)
from src.agents.prompts import (
    NEEDS_ALIGNMENT_PROMPT,
    KNOWLEDGE_TREE_PROMPT,
    CHAPTER_PLANNER_PROMPT,
    WRITER_PROMPT,
    REVIEWER_PROMPT,
    STRUCTURE_REVIEWER_PROMPT,
)
from src.schemas import (
    LearnerProfile, KnowledgeTree, ChapterPlan, ReviewResult,
    validate_or_raise,
)

def _this_year() -> str:
    return str(datetime.now().year)


# ================================================================
# Tool registry
# ================================================================

def _get_tool_definitions():
    """Return LangChain tool definitions for function calling."""
    from langchain_core.tools import tool
    from src.config import ENABLE_NETWORK_SEARCH

    tools = []

    if ENABLE_NETWORK_SEARCH:
        @tool
        def tavily_search(query: str, max_results: int = 10) -> str:
            """Search the web. Returns up to 10 articles with title, url, snippet.
            One search is usually enough — you get broad coverage in a single call.
            Use when you need up-to-date facts or real-world code examples.
            query: search keywords. max_results: 1-10 (default 10)."""
            from src.tools import tavily_search as _search
            results = _search(query, max_results)
            return json.dumps(results, ensure_ascii=False)
        tools.append(tavily_search)

    @tool
    def query_vector_store(query: str, top_k: int = 5) -> str:
        """Query local research knowledge base for previously saved research notes.
        Use when you want to reuse earlier research or check what was already found.
        query: semantic search query. top_k: number of results."""
        from src.tools import query_vector_store as _query
        docs = _query(query, top_k)
        return json.dumps(docs, ensure_ascii=False)
    tools.append(query_vector_store)

    return tuple(tools)


# ================================================================
# Function calling helper
# ================================================================

def _run_with_tools(llm, system_prompt: str, user_prompt: str, tool_list: list,
                    agent_name: str = "unknown") -> str:
    """Invoke LLM with function calling loop. Returns final text output.

    Agent-specific limits come from config.py:
      - MAX_TOOL_ROUNDS_PER_AGENT  → max LLM invocations (tool-call loop)
      - MAX_TOOL_CALLS_PER_AGENT   → max total tool executions (hard limit)
      - MAX_TOOL_SEC_PER_AGENT     → cumulative wall-clock timeout

    Flow:
      1. Bind tools → SystemMessage + HumanMessage
      2. For each round (up to max_rounds):
         a. Check remaining time (< 5s → break)
         b. LLM call (via trace_llm_call for monitoring)
         c. If no tool_calls → return content
         d. Execute each tool_call, append ToolMessage
         e. If hard limit reached → force LLM to output, don't call more tools
      3. Final fallback call → return content or timeout message
    """
    from langchain_core.messages import ToolMessage
    from src.monitor import get_active_tracer, trace_llm_call, trace_tool_call

    # Debug: log writer prompts to file for inspection
    if agent_name == "writer_chapter":
        from pathlib import Path
        debug_file = Path(__file__).parent.parent.parent / "data" / "writer_prompt_debug.txt"
        with open(debug_file, "w", encoding="utf-8") as df:
            df.write(f"=== SYSTEM PROMPT ({len(system_prompt)} chars) ===\n")
            df.write(system_prompt)
            df.write(f"\n\n=== USER PROMPT ({len(user_prompt)} chars) ===\n")
            df.write(user_prompt)
            df.write(f"\n\n=== TOOLS: {len(tool_list)} tools ===\n")
            for t in tool_list:
                df.write(f"  {t.name}: {t.description[:200] if hasattr(t, 'description') else 'N/A'}\n")

    max_rounds = MAX_TOOL_ROUNDS_PER_AGENT.get(agent_name, 2)
    max_tool_calls = MAX_TOOL_CALLS_PER_AGENT.get(agent_name, 4)
    max_tool_sec = MAX_TOOL_SEC_PER_AGENT.get(agent_name, 90)
    t_start = time.time()
    total_tool_calls = 0
    llm_with_tools = llm.bind_tools(tool_list)
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

    for round_idx in range(max_rounds):
        # Cumulative timeout: check before starting a new LLM call
        elapsed = time.time() - t_start
        remaining = max_tool_sec - elapsed
        if remaining <= 5:
            messages.append(HumanMessage(content=f"剩余时间不足({remaining:.0f}s)，请基于已有信息直接输出最终结果。"))
            break

        resp = trace_llm_call(
            lambda msgs: llm_with_tools.invoke(msgs), messages,
            model=getattr(llm, 'model_name', 'unknown'),
        )

        if not resp.tool_calls:
            return resp.content

        # Append AIMessage (with tool_calls) before its ToolMessages
        messages.append(resp)

        limit_reached = False
        for tc in resp.tool_calls:
            # Hard limit: skip tool execution but MUST send ToolMessage (API requirement)
            if total_tool_calls >= max_tool_calls:
                limit_reached = True
                messages.append(ToolMessage(
                    content=f"工具调用已达硬限制({max_tool_calls}次)，此调用被跳过。",
                    tool_call_id=tc.get("id", ""),
                ))
                continue

            total_tool_calls += 1
            tool_name = tc.get("name", "")
            args = tc.get("args", {})
            t0 = time.time()
            try:
                if tool_name == "tavily_search":
                    from src.tools import tavily_search as _search
                    result = _search(**args)
                elif tool_name == "query_vector_store":
                    from src.tools import query_vector_store as _query
                    result = _query(**args)
                else:
                    result = f"Unknown tool: {tool_name}"
            except Exception as e:
                result = f"Tool error: {e}"
            tool_latency = (time.time() - t0) * 1000

            if get_active_tracer():
                trace_tool_call(get_active_tracer(), tool_name, args, result, tool_latency)

            # Truncate tool results to prevent context explosion
            tool_content = json.dumps(result, ensure_ascii=False) if isinstance(result, (list, dict)) else str(result)
            if len(tool_content) > 4000:
                tool_content = tool_content[:4000] + f"\n...(truncated, total {len(tool_content)} chars)"
            messages.append(ToolMessage(
                content=tool_content,
                tool_call_id=tc.get("id", ""),
            ))

        # Hard limit reached: force LLM to output, no more rounds
        if limit_reached:
            messages.append(HumanMessage(content=f"工具调用已达硬限制({max_tool_calls}次)。请立即基于已有信息输出最终结果，不要再尝试调用工具。"))
            if time.time() - t_start < max_tool_sec - 5:
                return trace_llm_call(
                    lambda msgs: llm_with_tools.invoke(msgs), messages,
                    model=getattr(llm, 'model_name', 'unknown'),
                ).content
            break  # Time almost up, fall through to final best-effort call

    # Final fallback: force LLM to output based on accumulated messages
    if time.time() - t_start < max_tool_sec - 5:
        return trace_llm_call(
            lambda msgs: llm_with_tools.invoke(msgs), messages,
            model=getattr(llm, 'model_name', 'unknown'),
        ).content
    raise RuntimeError(
        f"[{agent_name}] LLM调用超时：累计{max_tool_sec}s已耗尽"
    )


# ================================================================
# Helpers
# ================================================================

def _level_instruction(level: str) -> str:
    """Look up the DEPTH_RULES instruction string for a given learner level."""
    return DEPTH_RULES.get(level, DEPTH_RULES["beginner"])["instruction"]


def _build_depth_rules_text(level: str = "") -> str:
    """Build a Markdown list of all DEPTH_RULES for prompt injection.

    If level is provided and valid, returns only that one rule.
    Otherwise returns all rules as a bullet list.
    """
    if level and level in DEPTH_RULES:
        rule = DEPTH_RULES[level]
        return f"- **{rule['label']}**：{rule['instruction']}"
    lines = []
    for key, rule in DEPTH_RULES.items():
        lines.append(f"- **{rule['label']}**：{rule['instruction']}")
    return "\n".join(lines)


def _chat_history_messages(history: list) -> list:
    """Convert internal message dicts OR LangChain objects to LangChain messages."""
    msgs = []
    for m in history:
        # LangGraph checkpointer may return LangChain objects, not dicts
        if hasattr(m, "content") and hasattr(m, "type"):
            # Already a LangChain message object — pass through
            msgs.append(m)
            continue
        if isinstance(m, dict):
            content = m.get("content", "")
            role = m.get("role", "")
        else:
            content = getattr(m, "content", "")
            role = getattr(m, "role", "")
        if role == "user" or role == "human":
            msgs.append(HumanMessage(content=content))
        elif role == "assistant" or role == "ai":
            msgs.append(AIMessage(content=content))
        elif role == "tool":
            msgs.append(ToolMessage(content=content, tool_call_id=m.get("tool_call_id", "")))
        elif role == "system":
            msgs.append(SystemMessage(content=content))
        else:
            msgs.append(HumanMessage(content=content))
    return msgs


def _normalize_level(level_raw: str) -> str:
    from src.schemas import normalize_level_str
    return normalize_level_str(level_raw)


# ================================================================
# Agent 1: NeedsAlignment ✅
# ================================================================
# (kept from previous version, unchanged)

def needs_alignment_node(state: dict) -> dict:
    """Structured info collection → detect completeness → follow-up or done.

    Greeting is shown directly by the UI — this node only handles post-greeting logic.
    """
    history = state.get("messages", [])

    if not history:
        return {"stage": "needs_alignment"}  # Let UI show greeting

    llm = get_llm(temperature=0.3)
    messages = [SystemMessage(content=NEEDS_ALIGNMENT_PROMPT), *_chat_history_messages(history)]
    resp = invoke_with_retry(llm, messages)
    profile = safe_extract_json(resp.content)

    has_required = (
        profile
        and profile.get("domain", "").strip()
        and profile.get("level", "").strip()
        and profile.get("goal", "").strip()
    )

    if has_required:
        profile["level"] = _normalize_level(profile.get("level", "beginner"))
        profile.setdefault("time_constraint", None)
        profile.setdefault("style", "balanced")
        try:
            validated = validate_or_raise(LearnerProfile, profile, "NeedsAlignment")
        except ValueError:
            validated = profile
        return {
            "user_needs": validated,
            "messages": history + [{"role": "assistant", "content": resp.content}],
            "stage": "needs_alignment_done",
        }

    return {
        "messages": history + [{"role": "assistant", "content": resp.content}],
        "stage": "needs_alignment",
    }


# ================================================================
# Agent 2: KnowledgeTreeBuilder
# ================================================================

def _parse_knowledge_tree_markdown(text: str) -> dict:
    """Parse Markdown topic list into KnowledgeTree dict.

    Expected format (pure list — one topic per line):

        # 学习领域
        - 知识点1
        - 知识点2
    """
    topics = []
    domain = ""

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("# ") and not stripped.startswith("## "):
            domain = stripped[2:].strip()
            continue
        if stripped.startswith("- ") or stripped.startswith("* "):
            topic = stripped[2:].strip()
            if topic:
                topics.append(topic)
            continue

    return {"domain": domain, "topics": topics}


def knowledge_tree_node(state: dict) -> dict:
    """Research domain → build 3-level knowledge tree (flat list + parent_id).

    Uses function calling: LLM searches, then outputs Markdown hierarchy.
    """
    llm = get_fast_llm(temperature=0.3)
    tools = list(_get_tool_definitions())
    profile = state.get("user_needs", {})

    level = profile.get("level", "beginner")
    level_label = DEPTH_RULES.get(level, DEPTH_RULES["beginner"])["label"]
    instruction = DEPTH_RULES.get(level, DEPTH_RULES["beginner"])["instruction"]
    system_prompt = KNOWLEDGE_TREE_PROMPT.replace("{depth_rules}", instruction)

    user_prompt = (
        f"## 学习者信息\n"
        f"- 学习领域：{profile.get('domain', '')}\n"
        f"- 当前水平：{level}（{level_label}）\n"
        f"- 学习目标：{profile.get('goal', '')}\n"
        f"- 风格偏好：{profile.get('style', 'balanced')}\n"
        f"- 当前年份：{_this_year()}\n\n"
        f"请列出从{level_label}水平到达学习目标需要掌握的所有知识点，按学习顺序排列。"
    )

    resp_text = _run_with_tools(llm, system_prompt, user_prompt, tools, agent_name="knowledge_tree")
    tree = _parse_knowledge_tree_markdown(resp_text)

    try:
        tree = validate_or_raise(KnowledgeTree, tree, "KnowledgeTreeBuilder")
    except ValueError:
        retry_prompt = (
            f"你之前的输出格式有误。请严格按格式重新输出：\n"
            f"# 领域名称  →  - 知识点1  →  - 知识点2\n"
            f"每行一个 - 知识点，不要分组，不要编号，不要标题。"
        )
        retry_text = _run_with_tools(llm, system_prompt, retry_prompt, tools, agent_name="knowledge_tree_retry")
        tree = _parse_knowledge_tree_markdown(retry_text)
        try:
            tree = validate_or_raise(KnowledgeTree, tree, "KnowledgeTreeBuilder-retry")
        except ValueError:
            # Retry failed → use whatever topics were parsed, domain from original parse
            parsed_domain = tree.get("domain", "") if isinstance(tree, dict) else ""
            tree = {"domain": parsed_domain, "topics": tree.get("topics", []) if isinstance(tree, dict) else []}

    return {"knowledge_tree": tree, "stage": "knowledge_tree_done"}


# ================================================================
# Agent 3: ChapterPlanner
# ================================================================

def _parse_chapter_markdown(text: str) -> dict:
    """Parse Markdown chapter plan into ChapterPlan dict.

    Industry pattern: ChapterPlanner only groups topics into chapters.
    Writer owns narrative structure (core questions, analogies, code).

    Expected format:

        # 博客标题

        ## 章节标题
        - 知识点1
        - 知识点2

        ## 章节标题
        - 知识点3
    """
    chapters = []
    current_chapter = None
    post_title = ""

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("# ") and not stripped.startswith("## "):
            post_title = stripped[2:].strip()
            continue

        if stripped.startswith("## "):
            if current_chapter:
                chapters.append(current_chapter)
            title = stripped[3:].strip().rstrip("：").rstrip(":")
            current_chapter = {"title": title, "key_points": []}
            continue

        if current_chapter is not None:
            if stripped.startswith("- ") or stripped.startswith("* "):
                point = stripped[2:].strip()
                if point:
                    current_chapter["key_points"].append(point)
                continue

    if current_chapter:
        chapters.append(current_chapter)

    return {"post_title": post_title, "chapters": chapters}


def split_chapters_by_budget(chapters: list[dict], budget: int, words_per_topic: int = 200) -> list[dict]:
    """Recursive binary-split: group chapters into posts within word budget.

    Uses chapter boundaries (natural semantic breaks from ChapterPlanner).
    Splits at chapter mid-points, recursing until each group fits in budget.
    Merges trailing posts that are too small (< 30% of budget).
    """
    if not chapters:
        return []

    total = sum(len(ch.get("key_points", [])) for ch in chapters) * words_per_topic

    # Base case: fits in budget or only 1 chapter
    if total <= budget or len(chapters) == 1:
        # Derive title from first chapter
        title = chapters[0].get("title", "")
        return [{
            "title": title,
            "chapter_indices": [],  # filled by caller
            "chapter_count": len(chapters),
            "word_estimate": total,
            "_chapters": chapters,  # temp, caller strips this
        }]

    # Split in half
    mid = len(chapters) // 2
    left = split_chapters_by_budget(chapters[:mid], budget, words_per_topic)
    right = split_chapters_by_budget(chapters[mid:], budget, words_per_topic)
    result = left + right

    # Merge trailing fragment: if last post < 30% budget, merge with previous
    if len(result) >= 2 and result[-1]["word_estimate"] < budget * 0.3:
        merged_chapters = result[-2]["_chapters"] + result[-1]["_chapters"]
        merged_total = sum(len(ch.get("key_points", [])) for ch in merged_chapters) * words_per_topic
        if merged_total <= budget:
            title = result[-2]["title"]
            result[-2] = {
                "title": title,
                "chapter_indices": [],
                "chapter_count": len(merged_chapters),
                "word_estimate": merged_total,
                "_chapters": merged_chapters,
            }
            result.pop()

    return result


def chapter_planner_node(state: dict) -> dict:
    """Plan ALL chapters from knowledge tree, then split into posts by budget.

    1. Takes ALL knowledge_tree topics → plans full chapter list
    2. Calls split_chapters_by_budget to divide chapters into posts
    3. Returns master chapter_plan + posts array with chapter_indices

    No longer per-post — runs once, output drives the entire multi-post loop.
    """
    llm = get_fast_llm(temperature=0.3)
    tools = list(_get_tool_definitions())
    profile = state.get("user_needs", {})

    # Get ALL topics
    tree = state.get("knowledge_tree", {})
    all_topics = tree.get("topics", [])
    domain = tree.get("domain", "")

    if not all_topics:
        return {"_error": "No topics to plan", "stage": "chapter_planner_error"}

    topics_text = "\n".join(f"- {t}" for t in all_topics)

    user_prompt = (
        f"## 学习领域：{domain}\n"
        f"用户水平：{profile.get('level', 'beginner')}\n\n"
        f"## 全部知识点（共{len(all_topics)}个）\n{topics_text}\n\n"
        f"将以上全部知识点按内容关联度分组为章节。关联紧密的放同一章，"
        f"在语义分界处换章。每章2-5个知识点。"
    )

    resp_text = _run_with_tools(llm, CHAPTER_PLANNER_PROMPT, user_prompt, tools, agent_name="chapter_planner")
    plan = _parse_chapter_markdown(resp_text)
    plan = validate_or_raise(ChapterPlan, plan, "ChapterPlanner")
    chapters = plan.get("chapters", [])

    # Split chapters into posts by word budget
    rule = DEPTH_RULES.get(profile.get("level", "beginner"), DEPTH_RULES["beginner"])
    post_budget = rule.get("max_words_per_chapter", 1200) * 5
    raw_posts = split_chapters_by_budget(chapters, post_budget)

    # Build post list: assign chapter_indices from master plan
    posts = []
    offset = 0
    for rp in raw_posts:
        ch_count = rp["chapter_count"]
        posts.append({
            "title": rp["title"],
            "chapter_indices": list(range(offset, offset + ch_count)),
            "chapter_count": ch_count,
            "word_estimate": rp["word_estimate"],
        })
        offset += ch_count

    if not posts:
        return {"_error": "Split produced no posts", "stage": "chapter_planner_error"}

    return {
        "chapter_plan": {"post_title": plan.get("post_title", domain), "chapters": chapters},
        "posts": posts,
        "current_post_index": 0,
        "current_post_title": posts[0].get("title", ""),
        "stage": "chapter_plan_done",
    }


# ================================================================
# Agent 4: Writer
# ================================================================

def writer_single_chapter_node(state: dict) -> dict:
    """Write ONE chapter in fan-out mode. Invoked once per chapter via Send().

    Reads _fanout_chapter_index (injected by Send) to determine which chapter
    to write. If review_feedback exists, enters fix-mode for that chapter.

    Returns per_chapter_drafts accumulator entry — LangGraph merges parallel
    results via operator.add reducer.
    """
    llm = get_fast_llm(temperature=0.5)  # Flash: 生成长文差距仅1.9分，12x成本节省
    tools = list(_get_tool_definitions())
    profile = state.get("user_needs", {})

    chapter_idx = state.get("_fanout_chapter_index", 0)
    chapters = state.get("chapter_plan", {}).get("chapters", [])
    if chapter_idx >= len(chapters):
        return {"per_chapter_drafts": []}

    chapter = chapters[chapter_idx]
    chapter_title = chapter.get("title", f"Chapter {chapter_idx + 1}")
    key_points = chapter.get("key_points", [])

    # Per-chapter word budget from DEPTH_RULES
    level = profile.get("level", "beginner")
    rule = DEPTH_RULES.get(level, DEPTH_RULES["beginner"])
    max_words = rule.get("max_words_per_chapter", 1200)
    # Strong word budget — repeated at start and end of prompt
    word_budget = (
        f"【硬性约束】本章最多 {max_words} 字。超过此限制会被自动打回重写。"
        f"每段控制在 200 字以内，代码块 ≤30 行。"
    )

    level_prompt = WRITER_PROMPT.replace(
        "{level_instruction}", _level_instruction(level)
    )

    review_feedback = state.get("review_feedback")
    reject_level = state.get("reject_level", "")
    if review_feedback is not None:
        # Fix mode: strategy depends on which review tier rejected
        chapter_contents = review_feedback.get("chapter_contents", {})
        original = chapter_contents.get(chapter_idx, "")
        issues = review_feedback.get("issues", [])
        ch_issues = [i for i in issues if i.get("chapter_index") == chapter_idx]
        issues_text = json.dumps(ch_issues, ensure_ascii=False, indent=2)

        if reject_level == "tier1":
            user_prompt = (
                f"## 章节：{chapter_title}\n"
                f"## 需要覆盖的知识点\n" + "\n".join(f"- {kp}" for kp in key_points) + "\n\n"
                f"## 字数约束：{word_budget}\n\n"
                f"## 原文\n{original}\n\n"
                f"## 代码检查发现问题（只修这些问题，其余原文照搬）\n{issues_text}\n\n"
                f"## 用户水平：{profile.get('level', 'beginner')}\n\n"
                f"修改规则：\n"
                f"1. 缺失的知识点 → 在合适位置插入简短讲解\n"
                f"2. 代码块超过30行 → 精简伪代码，移除样板\n"
                f"3. 字数超标 → 删减冗余内容，保留核心\n"
                f"4. 其他段落逐字保留原文，不要改动。输出完整章节。"
            )
        elif reject_level == "tier2":
            user_prompt = (
                f"## 章节：{chapter_title}\n"
                f"## 需要覆盖的知识点\n" + "\n".join(f"- {kp}" for kp in key_points) + "\n\n"
                f"## 字数约束：{word_budget}\n\n"
                f"## 原文\n{original}\n\n"
                f"## 结构审查意见\n{issues_text}\n\n"
                f"## 用户水平：{profile.get('level', 'beginner')}\n\n"
                f"修改规则：\n"
                f"1. 根据意见调整章节内段落顺序或过渡衔接\n"
                f"2. 如果意见涉及跨章调整（如拆分/合并），只修改本章\n"
                f"3. 段落内容保持不变，仅调整组织方式\n"
                f"4. 输出完整章节。"
            )
        else:
            user_prompt = (
                f"## 章节：{chapter_title}\n"
                f"## 需要覆盖的知识点\n" + "\n".join(f"- {kp}" for kp in key_points) + "\n\n"
                f"## 字数约束：{word_budget}\n\n"
                f"## 原文（逐段对照，只修改标注段落）\n{original}\n\n"
                f"## Reviewer 意见\n{issues_text}\n\n"
                f"## 用户水平：{profile.get('level', 'beginner')}\n\n"
                f"修改规则：对于标注了问题的段落，按建议逐段修改。"
                f"其他段落逐字保留原文。输出完整章节。"
            )
    else:
        user_prompt = (
            f"{word_budget}\n\n"
            f"## 章节：{chapter_title}\n"
            f"## 需要覆盖的知识点\n" + "\n".join(f"- {kp}" for kp in key_points) + "\n\n"
            f"## 用户水平：{profile.get('level', 'beginner')}\n"
            f"## 风格偏好：{profile.get('style', 'balanced')}\n\n"
            f"请只撰写这一章的内容（不要写整篇博客）。输出完整 Markdown 章节，"
            f"包含代码示例和运行结果。再次强调：{word_budget}"
        )

    draft = _run_with_tools(llm, level_prompt, user_prompt, tools, agent_name="writer_chapter")

    return {
        "per_chapter_drafts": [{
            "chapter_index": chapter_idx,
            "chapter_title": chapter_title,
            "draft_content": draft,
        }]
    }


def assembler_node(state: dict) -> dict:
    """Concatenate per-chapter drafts into a single blog post.

    On retry (partial update): fills missing chapters from the old assembled_draft.
    Sorts by chapter_index, prepends post title heading.
    """
    per_chapter = state.get("per_chapter_drafts", [])
    if not per_chapter:
        import logging
        log = logging.getLogger("BlogGen")
        log.error(f"ASSEMBLER: empty per_chapter_drafts. chapters={len(state.get('chapter_plan', {}).get('chapters', []))}, assembled_draft_len={len(state.get('assembled_draft', ''))}, draft_len={len(state.get('draft', ''))}")
        return {"_error": "No chapter drafts to assemble", "stage": "assembler_error"}

    chapter_plan = state.get("chapter_plan", {})
    chapters = chapter_plan.get("chapters", [])
    post_title = chapter_plan.get("post_title", state.get("current_post_title", ""))
    old_assembled = state.get("assembled_draft", state.get("draft", ""))

    # Map: chapter_index → new draft_content
    new_drafts = {d["chapter_index"]: d for d in per_chapter}

    parts = []
    if post_title:
        parts.append(f"# {post_title}\n")
    for i, ch in enumerate(chapters):
        if i in new_drafts:
            parts.append(new_drafts[i].get("draft_content", ""))
        else:
            # Retry skipped this chapter — reuse from old assembled draft
            fallback = _extract_chapter_draft(old_assembled, ch.get("title", ""))
            parts.append(fallback if fallback else f"## {ch.get('title', '')}\n\n(内容缺失)")
        parts.append("")

    assembled = "\n".join(parts).strip()

    retries = state.get("writer_retry_count", 0)
    return {
        "assembled_draft": assembled,
        "draft": assembled,
        "per_chapter_drafts": [],
        "writer_retry_count": retries,
        "stage": "writer_done",
    }


def writer_batch_node(state: dict) -> dict:
    """Prepare state for fan-out. Conditional edge route_writer_to_chapters
    returns [Send("write_chapter", ...)] → parallel writes → assembler."""
    review_fb = state.get("review_feedback")
    retries = state.get("writer_retry_count", 0) + (1 if review_fb else 0)
    return {
        "writer_retry_count": retries,
        "per_chapter_drafts": [],  # Reset accumulator for add reducer
    }


# ================================================================
# Review: Tier1 code-level check (no LLM, free)
# ================================================================

def tier1_check_node(state: dict) -> dict:
    """Code-level rule check on all chapters. No LLM calls — pure Python.

    Checks per chapter:
      1. Topic coverage: does each key_point appear in chapter content?
      2. Word count: is the chapter within DEPTH_RULES limits?
      3. Code block size: any code block >15 lines? (pseudo-code policy)

    Returns {tier1_pass: bool, tier1_issues: list}.
    If pass → routes to Tier2 (structure review).
    If fail → routes back to Writer with reject_level="tier1".
    """
    import re
    chapter_plan = state.get("chapter_plan", {})
    chapters = chapter_plan.get("chapters", [])
    assembled = state.get("assembled_draft", state.get("draft", ""))
    profile = state.get("user_needs", {})
    level = profile.get("level", "beginner")
    max_words = DEPTH_RULES.get(level, DEPTH_RULES["beginner"]).get("max_words_per_chapter", 2000)

    all_issues = []

    for i, ch in enumerate(chapters):
        ch_title = ch.get("title", f"Ch{i+1}")
        ch_content = _extract_chapter_draft(assembled, ch_title)
        if not ch_content:
            all_issues.append({
                "chapter_index": i,
                "paragraph": ch_title,
                "type": "知识点覆盖",
                "severity": "critical",
                "description": f"章节「{ch_title}」内容为空或未找到",
                "suggestion": "重新生成该章节",
            })
            continue

        # Check 1: topic coverage
        coverage_issues = _check_topic_coverage(ch, ch_content)
        for ci in coverage_issues:
            ci["chapter_index"] = i
        all_issues.extend(coverage_issues)

        # Check 2: word count per chapter
        ch_words = len(ch_content)
        if ch_words > max_words:
            all_issues.append({
                "chapter_index": i,
                "paragraph": ch_title,
                "type": "字数控制",
                "severity": "minor",
                "description": f"章节「{ch_title}」{ch_words}字，超过上限{max_words}字",
                "suggestion": f"精简内容至{max_words}字以内，优先保留核心原理和代码示例",
            })

        # Check 3: code block line count
        code_blocks = re.findall(r"```[\s\S]*?```", ch_content)
        for j, block in enumerate(code_blocks):
            lines = block.split("\n")
            code_lines = [l for l in lines if l.strip() and not l.strip().startswith("```")]
            if len(code_lines) > 30:
                all_issues.append({
                    "chapter_index": i,
                    "paragraph": f"第{j+1}个代码块",
                    "type": "代码示例质量",
                    "severity": "minor",
                    "description": f"代码块{j+1}有{len(code_lines)}行，超过30行限制",
                    "suggestion": "精简为简短伪代码（≤30行），移除import/异常处理/配置等样板，保留核心逻辑和运行结果",
                })

    if all_issues:
        # Collect chapter contents so Writer can do precise edits in fix-mode
        chapter_contents = {}
        for i, ch in enumerate(chapters):
            ch_title = ch.get("title", f"Ch{i+1}")
            ch_content = _extract_chapter_draft(assembled, ch_title)
            if ch_content:
                chapter_contents[i] = ch_content

        return {
            "tier1_pass": False,
            "review_result": {
                "action": "reject",
                "overall_assessment": f"Tier1代码检查：{len(all_issues)}个问题",
                "issues": all_issues,
            },
            "review_feedback": {
                "action": "reject",
                "issues": all_issues,
                "chapter_contents": chapter_contents,
            },
            "reject_level": "tier1",
            "stage": "review_reject",
        }

    return {"tier1_pass": True}


def prepare_review_batch_node(state: dict) -> dict:
    """Run structure review, then prepare for fan-out.

    Conditional edge route_review_to_chapters returns
    [Send("review_chapter", ...)] → parallel reviews → assemble_reviews.
    """
    chapters = state.get("chapter_plan", {}).get("chapters", [])
    if not chapters:
        return {"_error": "No chapters to review", "stage": "review_error"}

    struct_result = structure_reviewer_node(state)
    return {
        "structure_review": struct_result.get("structure_review", {}),
        "per_chapter_reviews": [],  # Reset accumulator for add reducer
    }


def prepare_fanout_node(state: dict) -> dict:
    """Reset per-chapter state before spawning fan-out writers.

    Increments writer_retry_count so the retry limit (MAX_REVIEW_RETRIES)
    is enforced across the full retry cycle, not per-chapter.
    Used only in Send-based fan-out mode (alternative to writer_batch_node).
    """
    current_retries = state.get("writer_retry_count", 0)
    return {
        "per_chapter_drafts": [],
        "writer_retry_count": current_retries + 1,
        "stage": "chapter_plan_done",
    }


# ================================================================
# Agent 5: Reviewer
# ================================================================

def _parse_review_markdown(text: str) -> dict:
    """Parse Markdown review output into ReviewResult dict.

    Expected format:

        判断：通过
        字数：12000
        总评：内容完整

        ---

        判断：不通过
        字数：19000
        总评：字数超标

        ### 问题1
        段落：全文
        类型：字数超标
        严重度：critical
        描述：超过18000字
        建议：按语义拆分

        #### 拆分建议
        原因：pipeline和架构选型之间分界
        第一篇：RAG核心流程 | 从开头到检索实战 | 完整pipeline
        第二篇：RAG进阶 | 向量选型到结尾 | 架构层面

    --- separator starts a new review (split-draft mode).
    Issue fields are parsed by prefix matching. Split suggestion is a #### subsection.
    """
    import re
    result = {"action": "accept", "word_count": 0, "overall_assessment": "", "issues": []}
    all_results = []  # for multiple reviews separated by ---
    current_issue = None
    in_split = False
    current_split = None

    def _finish_issue():
        nonlocal current_issue, in_split, current_split
        if current_issue:
            if current_split:
                current_issue["split_suggestion"] = current_split
            result["issues"].append(current_issue)
            current_issue = None
            in_split = False
            current_split = None

    def _finish_review():
        _finish_issue()
        r = dict(result)
        r["issues"] = list(result["issues"])  # copy
        all_results.append(r)
        result["action"] = "accept"
        result["word_count"] = 0
        result["overall_assessment"] = ""
        result["issues"] = []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        # --- separator → finish current review, start new one
        if stripped == "---" or stripped == "---":
            _finish_review()
            continue

        # ### problem header
        if stripped.startswith("### ") and not stripped.startswith("#### "):
            _finish_issue()
            current_issue = {
                "paragraph": "",
                "type": "",
                "severity": "minor",
                "description": "",
                "suggestion": "",
            }
            continue

        # #### split suggestion subsection
        if stripped.startswith("#### "):
            in_split = True
            current_split = {"split_reason": "", "groups": []}
            continue

        # Top-level fields
        if current_issue is None and not in_split:
            if stripped.startswith("判断：") or stripped.lower().startswith("action:"):
                prefix_end = stripped.index("：") + 1 if "：" in stripped else stripped.index(":") + 1
                v = stripped[prefix_end:].strip()
                result["action"] = "accept" if v in ("通过", "accept", "accept") else "reject"
                continue
            if stripped.startswith("字数：") or stripped.lower().startswith("word_count:"):
                prefix_end = stripped.index("：") + 1 if "：" in stripped else stripped.index(":") + 1
                try:
                    result["word_count"] = int(stripped[prefix_end:].strip())
                except ValueError:
                    pass
                continue
            if stripped.startswith("总评：") or stripped.lower().startswith("overall_assessment:"):
                prefix_end = stripped.index("：") + 1 if "：" in stripped else stripped.index(":") + 1
                result["overall_assessment"] = stripped[prefix_end:].strip()
                continue
            continue

        # Issue fields
        if current_issue is not None and not in_split:
            for field, key in [("章节：", "chapter_index"), ("段落：", "paragraph"),
                               ("类型：", "type"), ("严重度：", "severity"),
                               ("描述：", "description"), ("建议：", "suggestion")]:
                if stripped.startswith(field):
                    prefix_end = stripped.index("：") + 1 if "：" in stripped else len(field)
                    val = stripped[prefix_end:].strip() if "：" in stripped else stripped[len(field):].strip()
                    if key == "chapter_index":
                        try:
                            current_issue[key] = int(val) - 1  # 1-based → 0-based
                        except ValueError:
                            pass
                    else:
                        current_issue[key] = val
                    break
            continue

        # Split suggestion fields
        if in_split and current_split is not None:
            if stripped.startswith("原因：") or stripped.lower().startswith("reason:"):
                prefix_end = stripped.index("：") + 1 if "：" in stripped else stripped.index(":") + 1
                current_split["split_reason"] = stripped[prefix_end:].strip()
                continue
            # Group entries: "第一篇：title | range | rationale" or "第一篇：title"
            group_match = re.match(r"^第[一二三四五六七八九十\d]+篇[：:]", stripped)
            if group_match:
                content = stripped[len(group_match.group()):].strip()
                parts = [p.strip() for p in content.split("|")]
                group = {
                    "title": parts[0] if len(parts) > 0 else "",
                    "content_range": parts[1] if len(parts) > 1 else "",
                    "rationale": parts[2] if len(parts) > 2 else "",
                }
                current_split["groups"].append(group)
                continue

    _finish_issue()

    # If multiple reviews (split drafts), return the last one's structure
    # but the caller handles multiple reviews from the array
    if len(all_results) > 1:
        result = all_results[-1] if all_results else result
        # Attach all reviews for split-draft handling
        result["_all_reviews"] = all_results if all_results else [result]
    elif all_results:
        result = all_results[0]

    return result


# ================================================================
# Agent 5: Reviewer (fan-out per-chapter) + Review Assembler
# ================================================================

def _extract_chapter_draft(assembled: str, chapter_title: str) -> str:
    """Extract a single chapter's content from the assembled draft.

    Strategy 1: match "## {title}\n" → capture until next ## or EOF.
    Strategy 2 (fallback): split by all ## headings, pick by index.
    """
    if not assembled:
        return ""
    import re
    # Strategy 1: regex match by title
    pattern = rf"^##\s*{re.escape(chapter_title)}\s*$(.+?)(?=^##\s|\Z)"
    match = re.search(pattern, assembled, re.MULTILINE | re.DOTALL)
    if match:
        return match.group(1).strip()

    # Strategy 2: split by ## headings, match with normalized comparison
    sections = re.split(r"^##\s+", assembled, flags=re.MULTILINE)
    norm_title = chapter_title.strip().rstrip("：:").lower()
    for sec in sections[1:]:
        lines = sec.split("\n")
        heading = lines[0].strip().rstrip("：:").lower()
        # Exact match after normalization
        if heading == norm_title:
            return "\n".join(lines[1:]).strip()
        # Substring match: only if both are non-trivial length (avoid Ch1→Ch10)
        if len(norm_title) >= 4 and (norm_title in heading or heading in norm_title):
            return "\n".join(lines[1:]).strip()
    return ""


def _check_topic_coverage(chapter: dict, chapter_content: str) -> list[dict]:
    """Code-based check: which planned key_points are missing from the content?

    Returns issues for missing topics (no LLM needed).
    """
    issues = []
    for kp in chapter.get("key_points", []):
        # Extract core term: first segment before colon/comma
        core_term = kp.split("：")[0].split(":")[0].split("、")[0].strip()
        if not core_term:
            # Fallback: use full key_point if splitting yields empty
            core_term = kp.strip()
        if core_term and core_term not in chapter_content:
            issues.append({
                "chapter_index": -1,  # filled by caller
                "paragraph": f"知识点「{kp}」",
                "type": "知识点覆盖",
                "severity": "minor",
                "description": f"计划覆盖但正文中未找到：{kp}",
                "suggestion": f"在适当位置补充「{kp}」相关内容的讲解",
            })
    return issues


def _affected_chapters(review_feedback: dict) -> set[int]:
    """Extract chapter indices from review issues. Returns set of 0-based indices.

    If no chapter indices found (legacy review), returns all chapter indices.
    """
    issues = review_feedback.get("issues", [])
    indices = set()
    for issue in issues:
        ci = issue.get("chapter_index")
        if ci is not None and isinstance(ci, int):
            indices.add(ci)
    return indices if indices else None  # None = all chapters affected


reviewer_single_chapter_node: callable = None  # Forward decl, defined below


def _make_reviewer_single_chapter():
    """Factory: build the per-chapter checklist-based reviewer."""
    def _review(state: dict) -> dict:
        llm = get_llm(temperature=0.2)
        tools = []  # Reviewer uses checklist, no external tools needed
        profile = state.get("user_needs", {})

        chapter_idx = state.get("_review_chapter_index", 0)
        chapters = state.get("chapter_plan", {}).get("chapters", [])
        if chapter_idx >= len(chapters):
            return {"per_chapter_reviews": []}

        chapter = chapters[chapter_idx]
        chapter_title = chapter.get("title", f"Chapter {chapter_idx + 1}")

        assembled = state.get("assembled_draft", state.get("draft", ""))
        chapter_content = _extract_chapter_draft(assembled, chapter_title)
        if not chapter_content:
            chapter_content = f"(content not found for: {chapter_title})"

        key_points = chapter.get("key_points", [])
        points_text = "\n".join(f"- {kp}" for kp in key_points)

        # Code-based coverage check: which key_points are missing?
        # Code check on full content
        coverage_issues = _check_topic_coverage(chapter, chapter_content)
        ch_content_len = len(chapter_content)

        # Build checklist prompt
        level_str = _level_instruction(profile.get("level", "beginner"))
        ch_label = chapter_idx + 1  # 1-based for display

        sys_prompt = REVIEWER_PROMPT.replace("{key_points}", points_text)
        sys_prompt = sys_prompt.replace("{level_instruction}", level_str)
        sys_prompt = sys_prompt.replace("{chapter_idx}", str(ch_label))

        user_prompt = (
            f"## 章节 {ch_label}：{chapter_title}\n"
            f"## 正文（共 {ch_content_len} 字）\n{chapter_content}\n\n"
            f"如果「知识点覆盖」项不通过，请参考以下代码检测结果：\n"
            + "\n".join(f"- 缺失：{i['description']}" for i in coverage_issues)
        )

        resp_text = _run_with_tools(llm, sys_prompt, user_prompt, tools, agent_name="reviewer_chapter")
        review = _parse_review_markdown(resp_text)

        # Merge code-based coverage issues + LLM issues
        llm_issues = review.get("issues", [])
        for issue in llm_issues:
            if "chapter_index" not in issue:
                issue["chapter_index"] = chapter_idx
        # Add code-detected issues that LLM didn't already flag
        llm_descriptions = {i.get("description", "") for i in llm_issues}
        for ci in coverage_issues:
            if ci["description"] not in llm_descriptions:
                ci["chapter_index"] = chapter_idx
                llm_issues.append(ci)
        review["issues"] = llm_issues

        # Code-detected coverage gaps are hints for the LLM reviewer, not hard failures.
        # Substring matching can produce false positives when the Writer uses different
        # wording than the knowledge point list. The LLM reviewer makes the final call.
        # Only force reject if coverage issues are critical (e.g., empty chapter).
        critical_coverage = [i for i in coverage_issues if i.get("severity") == "critical"]
        if review.get("action") == "accept" and critical_coverage:
            review["action"] = "reject"
            review["overall_assessment"] = "关键内容缺失，需补充"

        try:
            review = validate_or_raise(ReviewResult, review, f"Reviewer-ch{chapter_idx}")
        except ValueError:
            pass

        return {
            "per_chapter_reviews": [{
                "chapter_index": chapter_idx,
                "review": review,
                "chapter_content": chapter_content,  # Full original → Writer fix-mode
            }],
        }
    return _review


reviewer_single_chapter_node = _make_reviewer_single_chapter()


def assemble_reviews_node(state: dict) -> dict:
    """Merge per-chapter reviews + structure review into single result.

    All pass → accept. Any chapter or structure fails → reject with merged issues.
    """
    per_chapter = state.get("per_chapter_reviews", [])
    structure_review = state.get("structure_review", {})
    if not per_chapter and not structure_review:
        return {"_error": "No reviews to assemble", "stage": "review_error"}

    sorted_reviews = sorted(per_chapter, key=lambda d: d.get("chapter_index", 0))
    all_accepted = True
    merged_issues = []
    chapter_contents = {}  # For Writer fix-mode: {chapter_index: original_content}
    total_words = 0

    for entry in sorted_reviews:
        review = entry.get("review", {})
        if review.get("action") != "accept":
            all_accepted = False
        merged_issues.extend(review.get("issues", []))
        total_words += review.get("word_count", 0)
        # Pass original chapter content for Writer's precise edit mode
        ci = entry.get("chapter_index")
        cc = entry.get("chapter_content")
        if ci is not None and cc is not None:
            chapter_contents[ci] = cc

    # Merge structure review
    if structure_review:
        if structure_review.get("action") != "accept":
            all_accepted = False
        merged_issues.extend(structure_review.get("issues", []))

    chapter_status = "、".join(
        f"第{entry['chapter_index']+1}章{'✓' if entry.get('review',{}).get('action')=='accept' else '✗'}"
        for entry in sorted_reviews
    )
    structure_status = "结构✓" if structure_review.get("action", "accept") == "accept" else "结构✗"
    overall = f"{structure_status} | {chapter_status}"

    merged = {
        "action": "accept" if all_accepted else "reject",
        "word_count": total_words if total_words > 0 else len(state.get("assembled_draft", "")),
        "overall_assessment": overall,
        "issues": merged_issues,
        "chapter_contents": chapter_contents,
    }

    if all_accepted:
        return {
            "review_result": merged,
            "review_feedback": {},
            "reject_level": "",
            "final": state.get("assembled_draft", state.get("draft", "")),
            "per_chapter_reviews": [],
            "structure_review": {},
            "stage": "review_pass",
        }
    else:
        # Structure failure is more fundamental → tier2; content only → tier3
        structure_failed = (
            structure_review.get("action") != "accept"
            if structure_review else False
        )
        return {
            "review_result": merged,
            "review_feedback": merged,
            "reject_level": "tier2" if structure_failed else "tier3",
            "per_chapter_reviews": [],
            "structure_review": {},
            "stage": "review_reject",
        }


def structure_reviewer_node(state: dict) -> dict:
    """Cross-chapter structural review: checks outline alignment + transitions.

    This runs ONCE per assembled draft (not per chapter). It checks things the
    per-chapter reviewers can't see:
    - Chapter ordering follows the intended progression
    - No content gaps between chapters
    - Post-level word count is within target
    - Topic coverage across all chapters is complete

    Uses checklist format, same as per-chapter reviewer.
    """
    llm = get_fast_llm(temperature=0.2)  # Flash: 结构审查是有明确清单的判别任务
    tools = []  # Structure review uses chapter summaries, no tools needed
    profile = state.get("user_needs", {})

    assembled = state.get("assembled_draft", state.get("draft", ""))
    chapter_plan = state.get("chapter_plan", {})
    chapters = chapter_plan.get("chapters", [])
    post_title = chapter_plan.get("post_title", state.get("current_post_title", ""))
    word_count = len(assembled)

    # Build a chapter index for the LLM
    chapter_list = "\n".join(
        f"{i+1}. {ch.get('title', f'Ch{i+1}')} — {'、'.join(ch.get('key_points', [])[:3])}"
        for i, ch in enumerate(chapters)
    )

    # Collect all key_points for completeness check
    all_points = []
    for ch in chapters:
        all_points.extend(ch.get("key_points", []))

    # Build chapter summaries: heading + first 120 chars of content + word count
    # This gives the structure reviewer enough context without sending full text
    chapter_summaries = []
    for i, ch in enumerate(chapters):
        ch_title = ch.get("title", f"Ch{i+1}")
        ch_content = _extract_chapter_draft(assembled, ch_title)
        summary = ch_content[:120].replace("\n", " ") if ch_content else "(内容缺失)"
        ch_word_count = len(ch_content) if ch_content else 0
        chapter_summaries.append(
            f"{i+1}. {ch_title} ({ch_word_count}字) — {summary}..."
        )

    # Inject dynamic context into the structure review template
    all_points_text = "\n".join(f"- {p}" for p in all_points)
    level_text = _level_instruction(profile.get("level", "beginner"))
    sys_prompt = STRUCTURE_REVIEWER_PROMPT.replace("{chapter_list}", chapter_list)
    sys_prompt = sys_prompt.replace("{all_points}", all_points_text)
    sys_prompt = sys_prompt.replace("{word_count}", str(word_count))
    sys_prompt = sys_prompt.replace("{level_instruction}", level_text)

    user_prompt = (
        f"## 博客：{post_title}\n"
        f"## 章节概要\n" + "\n".join(chapter_summaries) + "\n\n"
        f"## 全文字数：{word_count}\n"
        f"## 目标水平：{profile.get('level', 'beginner')}\n"
    )

    resp_text = _run_with_tools(llm, sys_prompt, user_prompt, tools, agent_name="structure_reviewer")
    structure_review = _parse_review_markdown(resp_text)

    return {
        "structure_review": structure_review,
    }
