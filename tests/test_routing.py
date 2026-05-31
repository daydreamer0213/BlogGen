"""Test graph routing functions: each route_after_* must return correct next node."""
import pytest
from src.graph.builder import (
    route_after_needs_alignment,
    route_after_knowledge_tree,
    route_after_chapter_planner,
    route_after_assembler,
    route_after_tier1,
    route_after_review,
)
from src.graph.state import initial_state, BlogGenState


class TestRouting:
    def test_route_needs_alignment_done(self):
        state = {**initial_state(), "stage": "needs_alignment_done"}
        assert route_after_needs_alignment(state) == "knowledge_tree"

    def test_route_needs_alignment_pending(self):
        state = {**initial_state(), "stage": "needs_alignment"}
        assert route_after_needs_alignment(state) == "needs_alignment"

    def test_route_knowledge_tree_done(self):
        state = {**initial_state(), "stage": "knowledge_tree_done"}
        assert route_after_knowledge_tree(state) == "chapter_planner"

    def test_route_knowledge_tree_not_done(self):
        state = {**initial_state(), "stage": "knowledge_tree"}
        assert route_after_knowledge_tree(state) == "__end__"

    def test_route_chapter_planner_done(self):
        state = {**initial_state(), "stage": "chapter_plan_done"}
        assert route_after_chapter_planner(state) == "writer_batch"

    def test_route_chapter_planner_not_done(self):
        state = {**initial_state(), "stage": "chapter_planner"}
        assert route_after_chapter_planner(state) == "__end__"

    def test_route_writer_done(self):
        state = {**initial_state(), "stage": "writer_done"}
        assert route_after_assembler(state) == "tier1_check"

    def test_route_writer_not_done(self):
        state = {**initial_state(), "stage": "writer_batch"}
        assert route_after_assembler(state) == "__end__"

    def test_route_tier1_pass(self):
        state = {**initial_state(), "tier1_pass": True}
        assert route_after_tier1(state) == "review_batch"

    def test_route_tier1_fail(self):
        state = {**initial_state(), "tier1_pass": False, "writer_retry_count": 0}
        assert route_after_tier1(state) == "writer_batch"

    def test_route_tier1_fail_but_retries_exhausted(self):
        """After max retries, tier1 failure still goes to human review."""
        state = {**initial_state(), "tier1_pass": False, "writer_retry_count": 3}
        assert route_after_tier1(state) == "review_batch"

    def test_route_tier1_default(self):
        state = {**initial_state()}
        assert route_after_tier1(state) == "writer_batch"

    def test_route_review_pass_first_post(self):
        state = {
            **initial_state(),
            "stage": "review_pass",
            "current_post_index": 0,
            "knowledge_tree": {"domain": "AI", "topics": ["RAG", "Agent", "Fine-tuning"]},
        }
        assert route_after_review(state) == "next_post"

    def test_route_review_pass_last_post(self):
        state = {
            **initial_state(),
            "stage": "review_pass",
            "current_post_index": 2,
            "knowledge_tree": {"domain": "AI", "topics": ["RAG", "Agent", "Fine-tuning"]},
        }
        assert route_after_review(state) == "__end__"

    def test_route_review_pass_single_topic(self):
        state = {
            **initial_state(),
            "stage": "review_pass",
            "current_post_index": 0,
            "knowledge_tree": {"domain": "AI", "topics": ["RAG"]},
        }
        assert route_after_review(state) == "__end__"

    def test_route_review_reject_within_retry_limit(self):
        state = {**initial_state(), "stage": "review_reject", "writer_retry_count": 0}
        assert route_after_review(state) == "writer_batch"

    def test_route_review_reject_first_retry(self):
        state = {**initial_state(), "stage": "review_reject", "writer_retry_count": 1}
        # MAX_REVIEW_RETRIES = 2, so retry_count=1 is < 2 → still retry
        from src.config import MAX_REVIEW_RETRIES
        if MAX_REVIEW_RETRIES > 1:
            assert route_after_review(state) == "writer_batch"

    def test_route_review_reject_exceeds_retry_limit(self):
        state = {**initial_state(), "stage": "review_reject", "writer_retry_count": 3}
        assert route_after_review(state) == "__end__"

    def test_route_review_unknown_stage_defaults_end(self):
        state = {**initial_state(), "stage": "unknown_stage"}
        assert route_after_review(state) == "__end__"
