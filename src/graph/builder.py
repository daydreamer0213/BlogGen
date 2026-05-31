"""LangGraph graph for 5-agent BlogGen pipeline with parallel Writer/Reviewer.

Uses Send() from conditional edges for fan-out (LangGraph 1.x pattern).
"""
from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Send

from src.graph.state import BlogGenState
from src.agents.nodes import (
    needs_alignment_node,
    knowledge_tree_node,
    chapter_planner_node,
    writer_batch_node,
    writer_single_chapter_node,
    assembler_node,
    tier1_check_node,
    prepare_review_batch_node,
    reviewer_single_chapter_node,
    assemble_reviews_node,
)
from src.config import MAX_REVIEW_RETRIES


# ============================================================
# Routing functions
# ============================================================

def route_after_needs_alignment(state: BlogGenState) -> Literal["knowledge_tree", "needs_alignment"]:
    if state.get("stage") == "needs_alignment_done":
        return "knowledge_tree"
    return "needs_alignment"


def route_after_knowledge_tree(state: BlogGenState) -> Literal["chapter_planner", "__end__"]:
    if state.get("stage") == "knowledge_tree_done":
        return "chapter_planner"
    return "__end__"


def route_after_chapter_planner(state: BlogGenState) -> Literal["writer_batch", "__end__"]:
    if state.get("stage") == "chapter_plan_done":
        return "writer_batch"
    return "__end__"


# ---- Writer fan-out: conditional edge returns Send() for parallel chapter writes ----


def route_writer_to_chapters(state: BlogGenState):
    """Fan-out: return [Send("write_chapter", ...)] for parallel chapter writes."""
    chapters = state.get("chapter_plan", {}).get("chapters", [])
    review_fb = state.get("review_feedback")
    posts = state.get("posts", [])
    idx = state.get("current_post_index", 0)

    if review_fb:
        from src.agents.nodes import _affected_chapters
        affected = _affected_chapters(review_fb)
        indices = [i for i in (affected or range(len(chapters))) if i < len(chapters)]
    elif posts and idx < len(posts) and "chapter_indices" in posts[idx]:
        indices = posts[idx]["chapter_indices"]
    else:
        indices = list(range(len(chapters)))

    if not indices:
        return "assembler"

    return [Send("write_chapter", {
        **{k: v for k, v in state.items() if not k.startswith("_")},
        "_fanout_chapter_index": i,
    }) for i in indices]


def route_after_assembler(state: BlogGenState) -> Literal["tier1_check", "__end__"]:
    if state.get("stage") == "writer_done":
        return "tier1_check"
    return "__end__"


# ---- Review fan-out: conditional edge returns Send() for parallel chapter reviews ----


def route_review_to_chapters(state: BlogGenState):
    """Fan-out: return [Send("review_chapter", ...)] for parallel chapter reviews."""
    chapters = state.get("chapter_plan", {}).get("chapters", [])
    if not chapters:
        return "assemble_reviews"
    # Structure review result is already in state (set by review_batch node)
    return [Send("review_chapter", {
        **{k: v for k, v in state.items() if not k.startswith("_")},
        "_review_chapter_index": i,
    }) for i in range(len(chapters))]


def route_after_tier1(state: BlogGenState) -> Literal["review_batch", "writer_batch"]:
    if state.get("tier1_pass", False):
        return "review_batch"
    if state.get("writer_retry_count", 0) > MAX_REVIEW_RETRIES:
        return "review_batch"
    return "writer_batch"


def route_after_review(state: BlogGenState) -> Literal["writer_batch", "next_post", END]:
    stage = state.get("stage")
    retries = state.get("writer_retry_count", 0)

    if stage == "review_reject" and retries <= MAX_REVIEW_RETRIES:
        return "writer_batch"

    if stage == "review_pass":
        current = state.get("current_post_index", 0)
        posts = state.get("posts", [])
        if posts and current + 1 < len(posts):
            return "next_post"
        kt_topics = state.get("knowledge_tree", {}).get("topics", [])
        if current + 1 < len(kt_topics):
            return "next_post"
        return END

    return END


# ============================================================
# Next post transition
# ============================================================

def next_post_node(state: BlogGenState) -> dict:
    current = state.get("current_post_index", 0)
    next_idx = current + 1
    posts = state.get("posts", [])

    if not posts:
        return {"stage": "done"}

    if next_idx < len(posts):
        return {
            "posts": posts,
            "current_post_index": next_idx,
            "current_post_title": posts[next_idx]["title"],
            "chapter_plan": state.get("chapter_plan", {}),
            "draft": "", "assembled_draft": "",
            "per_chapter_drafts": [], "per_chapter_reviews": [],
            "writer_retry_count": 0, "review_result": {}, "review_feedback": {},
            "final": "", "stage": "writer_batch",
        }
    return {"stage": "done"}


# ============================================================
# Build
# ============================================================

def build_graph() -> StateGraph:
    graph = StateGraph(BlogGenState)

    graph.add_node("needs_alignment", needs_alignment_node)
    graph.add_node("knowledge_tree", knowledge_tree_node)
    graph.add_node("chapter_planner", chapter_planner_node)
    graph.add_node("writer_batch", writer_batch_node)
    graph.add_node("write_chapter", writer_single_chapter_node)
    graph.add_node("assembler", assembler_node)
    graph.add_node("tier1_check", tier1_check_node)
    graph.add_node("review_batch", prepare_review_batch_node)
    graph.add_node("review_chapter", reviewer_single_chapter_node)
    graph.add_node("assemble_reviews", assemble_reviews_node)
    graph.add_node("next_post", next_post_node)

    graph.set_entry_point("needs_alignment")

    graph.add_conditional_edges("needs_alignment", route_after_needs_alignment, {
        "knowledge_tree": "knowledge_tree",
        "needs_alignment": "needs_alignment",
    })

    graph.add_conditional_edges("knowledge_tree", route_after_knowledge_tree, {
        "chapter_planner": "chapter_planner",
        "__end__": END,
    })

    graph.add_conditional_edges("chapter_planner", route_after_chapter_planner, {
        "writer_batch": "writer_batch",
        "__end__": END,
    })

    # Writer fan-out: writer_batch → Send("write_chapter") ×N → assembler
    graph.add_conditional_edges("writer_batch", route_writer_to_chapters, {
        "write_chapter": "write_chapter",
        "assembler": "assembler",
    })
    graph.add_edge("write_chapter", "assembler")
    graph.add_conditional_edges("assembler", route_after_assembler, {
        "tier1_check": "tier1_check",
        "__end__": END,
    })

    graph.add_conditional_edges("tier1_check", route_after_tier1, {
        "review_batch": "review_batch",
        "writer_batch": "writer_batch",
    })

    # Review fan-out: review_batch → Send("review_chapter") ×N → assemble_reviews
    graph.add_conditional_edges("review_batch", route_review_to_chapters, {
        "review_chapter": "review_chapter",
        "assemble_reviews": "assemble_reviews",
    })
    graph.add_edge("review_chapter", "assemble_reviews")
    graph.add_conditional_edges("assemble_reviews", route_after_review, {
        "writer_batch": "writer_batch",
        "next_post": "next_post",
        END: END,
    })

    graph.add_edge("next_post", "writer_batch")

    return graph


def compile_graph(interrupt_after: list[str] | None = None):
    return build_graph().compile(
        checkpointer=MemorySaver(),
        interrupt_after=interrupt_after or [],
    )
