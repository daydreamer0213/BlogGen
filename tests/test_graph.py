"""Graph-level integration tests for the LangGraph pipeline."""
import pytest
from unittest.mock import MagicMock, patch
from tests.conftest import needs_langgraph


@needs_langgraph
class TestGraphCompilation:
    def test_graph_compiles(self):
        from src.graph.builder import build_graph, compile_graph
        graph = build_graph()
        assert graph is not None
        compiled = compile_graph()
        assert compiled is not None

    def test_graph_compiles_with_interrupt(self):
        from src.graph.builder import compile_graph
        compiled = compile_graph(interrupt_after=["needs_alignment"])
        assert compiled is not None


@needs_langgraph
class TestRoutingFunctions:
    def test_route_after_chapter_planner(self):
        from src.graph.builder import route_after_chapter_planner
        assert route_after_chapter_planner({"stage": "chapter_plan_done"}) == "writer_batch"

    def test_route_after_review_retry(self):
        from src.graph.builder import route_after_review
        result = route_after_review({
            "stage": "review_reject", "writer_retry_count": 0,
            "current_post_index": 0, "knowledge_tree": {"topics": ["A"]},
        })
        assert result == "writer_batch"

    def test_route_after_review_accept_last(self):
        from src.graph.builder import route_after_review
        from langgraph.graph import END
        result = route_after_review({
            "stage": "review_pass", "current_post_index": 0,
            "knowledge_tree": {"topics": ["A"]},
        })
        assert result == END

    def test_route_after_review_accept_next(self):
        from src.graph.builder import route_after_review
        result = route_after_review({
            "stage": "review_pass", "current_post_index": 0,
            "knowledge_tree": {"topics": ["A", "B"]},
        })
        assert result == "next_post"

    def test_route_after_assembler(self):
        from src.graph.builder import route_after_assembler
        assert route_after_assembler({"stage": "writer_done"}) == "tier1_check"

    def test_route_after_tier1_pass(self):
        from src.graph.builder import route_after_tier1
        assert route_after_tier1({"tier1_pass": True}) == "review_batch"

    def test_route_after_tier1_fail(self):
        from src.graph.builder import route_after_tier1
        assert route_after_tier1({"tier1_pass": False, "writer_retry_count": 0}) == "writer_batch"

    def test_route_after_tier1_fail_retries_exhausted(self):
        from src.graph.builder import route_after_tier1
        assert route_after_tier1({"tier1_pass": False, "writer_retry_count": 3}) == "review_batch"


@needs_langgraph
class TestNextPostNode:
    def test_resets_per_post_fields(self):
        from src.graph.builder import next_post_node
        result = next_post_node({
            "current_post_index": 0,
            "posts": [{"title": "Post1", "topics": ["A"]}, {"title": "Post2", "topics": ["B"]}],
            "chapter_plan": {}, "draft": "old", "writer_retry_count": 2,
            "review_result": {}, "review_feedback": {}, "final": "f",
        })
        assert result["current_post_index"] == 1
        assert result["current_post_title"] == "Post2"
        assert result["writer_retry_count"] == 0
        assert result["stage"] == "writer_batch"


@needs_langgraph
class TestSession:
    def test_session_create(self):
        from src.graph.session import BlogGenSession
        session = BlogGenSession()
        state = session.create()
        assert state["stage"] == "needs_alignment"

    def test_is_interrupted_after_create(self):
        from src.graph.session import BlogGenSession
        session = BlogGenSession()
        session.create()
        assert session.is_interrupted()

    def test_update_state(self):
        from src.graph.session import BlogGenSession
        session = BlogGenSession()
        session.create()
        session.update_state({"stage": "test"})
        assert session.get_state()["stage"] == "test"
