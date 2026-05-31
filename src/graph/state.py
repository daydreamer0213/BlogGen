"""BlogGen state schema for LangGraph."""
from operator import add
from typing import TypedDict, Annotated, Optional
from langgraph.graph.message import add_messages
from datetime import datetime


class BlogGenState(TypedDict):
    # ── Messages (chat history for NeedsAlignment) ──
    messages: Annotated[list, add_messages]

    # ── Agent 1: NeedsAlignment ──
    user_needs: dict  # LearnerProfile

    # ── Agent 2: KnowledgeTreeBuilder ──
    knowledge_tree: dict  # KnowledgeTree: {domain, topics}

    # ── SplitPosts → Agent 3: ChapterPlanner (per-post) ──
    posts: list[dict]            # [{title, topics}] — split plan from SplitPosts
    current_post_index: int
    current_post_title: str
    current_post_topics: list[str]  # Deprecated: kept for backward compat
    chapter_plan: dict  # ChapterPlan: {post_title, chapters}

    # ── Fan-out Writer ──
    per_chapter_drafts: Annotated[list[dict], add]  # [{chapter_index, chapter_title, draft_content}]
    per_chapter_reviews: Annotated[list[dict], add]  # [{chapter_index, review}]
    assembled_draft: str  # Concatenated output from Assembler
    tier1_pass: bool            # Set by tier1_check_node, read by route_after_tier1

    # ── Agent 4: Writer ──
    draft: str             # Written Markdown
    writer_retry_count: int  # How many times Writer has been called for current post

    # ── Agent 5: Reviewer ──
    review_result: dict       # ReviewResult
    review_feedback: dict     # Feedback to Writer (if rejected)
    reject_level: str         # "tier1" | "tier2" | "tier3" — which review tier rejected
    final: str               # Final approved content

    # ── Blog post tracking ──
    completed_posts: list[dict]  # [{title, content, ...}]

    # ── Flow control ──
    stage: str

    # ── HITL flags ──
    needs_approved: bool
    tree_approved: bool
    chapter_plan_approved: bool
    final_approved: bool

    # ── Session metadata ──
    session_created_at: str


def initial_state() -> dict:
    return {
        "messages": [],
        "user_needs": {},
        "knowledge_tree": {},
        "posts": [],
        "current_post_index": 0,
        "current_post_title": "",
        "current_post_topics": [],
        "chapter_plan": {},
        "per_chapter_drafts": [],
        "per_chapter_reviews": [],
        "assembled_draft": "",
        "draft": "",
        "writer_retry_count": 0,
        "review_result": {},
        "review_feedback": {},
        "reject_level": "",
        "final": "",
        "completed_posts": [],
        "stage": "needs_alignment",
        "needs_approved": False,
        "tree_approved": False,
        "chapter_plan_approved": False,
        "final_approved": False,
        "session_created_at": datetime.now().isoformat(),
    }
