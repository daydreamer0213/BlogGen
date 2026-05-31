"""Integration tests: agent chains with mocked LLM responses.

Tests the interaction between components, function calling loops,
and multi-node workflows.
"""
import json
import pytest
from unittest.mock import MagicMock, patch, ANY
from tests.conftest import needs_all_deps, needs_bs4, needs_langgraph, needs_jinja2
import src.tools  # noqa: F401 — ensure module is importable for patch() resolution


# ================================================================
# _run_with_tools function calling loop
# ================================================================

class TestFunctionCallingLoop:
    """Test the _run_with_tools function calling mechanism."""

    def test_single_llm_call_no_tools(self):
        """LLM responds without calling any tools → direct text output."""
        from src.agents.nodes import _run_with_tools

        mock_resp = MagicMock()
        mock_resp.content = '{"domain": "AI", "level": "beginner", "goal": "test"}'
        mock_resp.tool_calls = []  # No tools called

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm  # bind_tools returns self
        mock_llm.invoke.return_value = mock_resp

        result = _run_with_tools(mock_llm, "sys prompt", "user prompt", [], agent_name="test")
        assert result == mock_resp.content

    def test_tool_call_with_result(self):
        """LLM calls tavily_search → receives result → responds."""
        from src.agents.nodes import _run_with_tools
        from unittest.mock import patch

        # Round 1: LLM calls a tool
        mock_tc = {
            "name": "tavily_search",
            "args": {"query": "RAG tutorial", "max_results": 5},
            "id": "call_1",
        }
        mock_resp1 = MagicMock()
        mock_resp1.tool_calls = [mock_tc]
        mock_resp1.content = ""

        # Round 2: LLM responds after getting tool result
        mock_resp2 = MagicMock()
        mock_resp2.content = '{"result": "found"}'
        mock_resp2.tool_calls = []

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [mock_resp1, mock_resp2]
        mock_llm.bind_tools.return_value = mock_llm  # bind_tools returns self

        with patch("src.monitor.get_active_tracer", return_value=None):
            with patch("src.tools.tavily_search") as mock_search:
                mock_search.return_value = [{"title": "RAG Tutorial", "url": "http://x", "snippet": "..."}]
                result = _run_with_tools(mock_llm, "sys", "user", [], agent_name="test")

        assert result == '{"result": "found"}'
        assert mock_search.call_count == 1
        assert mock_llm.invoke.call_count == 2  # tool call + final response

    def test_multiple_tool_rounds(self):
        """LLM calls tool → gets result → calls another tool → responds."""
        from src.agents.nodes import _run_with_tools
        from unittest.mock import patch

        tc1 = {
            "name": "tavily_search",
            "args": {"query": "first search", "max_results": 3},
            "id": "c1",
        }
        tc2 = {
            "name": "query_vector_store",
            "args": {"query": "second lookup", "top_k": 3},
            "id": "c2",
        }
        resp1 = MagicMock()
        resp1.tool_calls = [tc1]
        resp2 = MagicMock()
        resp2.tool_calls = [tc2]
        resp3 = MagicMock()
        resp3.tool_calls = []
        resp3.content = "final answer"

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [resp1, resp2, resp3]
        mock_llm.bind_tools.return_value = mock_llm

        with patch("src.monitor.get_active_tracer", return_value=None):
            with patch("src.tools.tavily_search", return_value=[{"title": "x"}]):
                with patch("src.tools.query_vector_store", return_value=[{"content": "y"}]):
                    result = _run_with_tools(mock_llm, "sys", "user", [], agent_name="test")

        assert result == "final answer"
        assert mock_llm.invoke.call_count == 3

    def test_max_rounds_prevent_loop(self):
        """LLM keeps calling tools → capped at max_rounds."""
        from src.agents.nodes import _run_with_tools
        from unittest.mock import patch

        tc = {"name": "tavily_search", "args": {"query": "loop"}, "id": "c1"}
        resp = MagicMock()
        resp.tool_calls = [tc]
        resp.content = "last resort"

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = resp  # Always returns tool call
        mock_llm.bind_tools.return_value = mock_llm

        with patch("src.monitor.get_active_tracer", return_value=None):
            with patch("src.tools.tavily_search", return_value=[]):
                result = _run_with_tools(mock_llm, "sys", "user", [], agent_name="test")

        # 2 rounds (both tool calls) + 1 final fallback = 3 total invoke calls
        assert mock_llm.invoke.call_count == 3
        assert result == "last resort"

    def test_tool_error_handled(self):
        """Tool execution fails → error returned to LLM → LLM continues."""
        from src.agents.nodes import _run_with_tools
        from unittest.mock import patch

        tc = {"name": "tavily_search", "args": {"query": "x"}, "id": "c1"}
        resp1 = MagicMock()
        resp1.tool_calls = [tc]
        resp2 = MagicMock()
        resp2.tool_calls = []
        resp2.content = "recovered despite tool error"

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [resp1, resp2]
        mock_llm.bind_tools.return_value = mock_llm

        with patch("src.monitor.get_active_tracer", return_value=None):
            with patch("src.tools.tavily_search", side_effect=Exception("API down")):
                result = _run_with_tools(mock_llm, "sys", "user", [], agent_name="test")

        # Tool error message is passed back as string result to LLM
        assert mock_llm.invoke.call_count == 2  # LLM got error + recovered
        assert result == "recovered despite tool error"

    def test_unknown_tool_handled(self):
        """LLM calls a tool we don't have → recorded as unknown."""
        from src.agents.nodes import _run_with_tools
        from unittest.mock import patch

        tc = {"name": "nonexistent_tool", "args": {}, "id": "c1"}
        resp1 = MagicMock()
        resp1.tool_calls = [tc]
        resp2 = MagicMock()
        resp2.tool_calls = []
        resp2.content = "done"

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [resp1, resp2]
        mock_llm.bind_tools.return_value = mock_llm

        with patch("src.monitor.get_active_tracer", return_value=None):
            result = _run_with_tools(mock_llm, "sys", "user", [], agent_name="test")

        assert result == "done"  # Doesn't crash on unknown tool


# ================================================================
# Agent chain integration
# ================================================================

class TestAgentChains:
    """Test multi-agent chains with mocked LLM."""

    def test_needs_alignment_to_knowledge_tree(self):
        """Full flow: alignment extraction → tree building."""
        from src.agents.nodes import needs_alignment_node

        # Simulate user input after greeting
        state = {
            "messages": [
                {"role": "assistant", "content": "请告诉我..."},
                {"role": "user", "content": "我想学RAG，初级水平，目标是面试准备"},
            ],
            "stage": "needs_alignment",
        }

        mock_resp = MagicMock()
        mock_resp.content = '{"domain": "RAG", "level": "beginner", "goal": "面试准备"}'

        with patch("src.agents.nodes.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = mock_resp
            mock_get_llm.return_value = mock_llm
            with patch("src.agents.nodes.invoke_with_retry", return_value=mock_resp):
                result = needs_alignment_node(state)

        assert result["stage"] == "needs_alignment_done"
        assert result["user_needs"]["domain"] == "RAG"
        assert result["user_needs"]["level"] == "beginner"

    def test_writer_fix_mode_full_cycle(self):
        """Writer single-chapter with review feedback → fix mode."""
        from src.agents.nodes import writer_single_chapter_node

        mock_llm = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = "# Fixed\n\nFixed paragraph."
        mock_llm.invoke.return_value = mock_resp

        feedback = {
            "action": "reject",
            "issues": [{"paragraph": "第3段", "type": "事实错误", "description": "err", "suggestion": "fix"}],
        }

        state = {
            "user_needs": {"domain": "AI", "level": "beginner", "goal": "面试", "style": "balanced"},
            "chapter_plan": {"chapters": [{"title": "Ch1", "key_points": ["a", "b"]}]},
            "_fanout_chapter_index": 0,
            "review_feedback": feedback,
        }

        with patch("src.agents.nodes.get_llm", return_value=mock_llm):
            with patch("src.agents.nodes._run_with_tools", return_value=mock_resp.content):
                result = writer_single_chapter_node(state)

        assert len(result["per_chapter_drafts"]) == 1
        assert "Fixed" in result["per_chapter_drafts"][0]["draft_content"]

    def test_writer_no_feedback_normal_mode(self):
        """Writer single-chapter without feedback → normal mode."""
        from src.agents.nodes import writer_single_chapter_node

        mock_llm = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = "# Chapter 1\n\nContent."
        mock_llm.invoke.return_value = mock_resp

        state = {
            "user_needs": {"domain": "AI", "level": "beginner", "goal": "面试", "style": "balanced"},
            "chapter_plan": {"chapters": [{"title": "Ch1", "key_points": ["a"]}]},
            "_fanout_chapter_index": 0,
            "review_feedback": None,
        }

        with patch("src.agents.nodes.get_llm", return_value=mock_llm):
            with patch("src.agents.nodes._run_with_tools", return_value=mock_resp.content):
                result = writer_single_chapter_node(state)

        assert result["per_chapter_drafts"][0]["chapter_index"] == 0

    def test_reviewer_accept_flow(self):
        """prepare_review_batch_node accepts → returns review_pass."""
        from src.agents.nodes import prepare_review_batch_node, structure_reviewer_node, reviewer_single_chapter_node, assemble_reviews_node

        mock_llm = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = "判断：通过\n字数：12000\n总评：good\n"
        mock_llm.invoke.return_value = mock_resp

        state = {
            "user_needs": {"domain": "AI", "level": "beginner", "goal": "面试"},
            "assembled_draft": "# Great blog\n\n## Ch1\n\nContent here.",
            "chapter_plan": {"chapters": [{"title": "Ch1", "key_points": ["kp1"]}]},
        }

        with patch("src.agents.nodes.get_fast_llm", return_value=mock_llm):
            with patch("src.agents.nodes.get_llm", return_value=mock_llm):
                with patch("src.agents.nodes._run_with_tools", return_value=mock_resp.content):
                    # Run review pipeline: structure + chapter + assemble (review_batch uses Send in production)
                    struct_r = structure_reviewer_node(state)
                    ch_r = reviewer_single_chapter_node({**state, "_review_chapter_index": 0})
                    all_r = ch_r.get("per_chapter_reviews", [])
                    result = assemble_reviews_node({**state, "per_chapter_reviews": all_r, "structure_review": struct_r.get("structure_review", {})})

        assert result["stage"] == "review_pass"


# ================================================================
# State machine transitions
# ================================================================

@needs_langgraph
class TestStateTransitions:
    def test_knowledge_tree_routing(self):
        from src.graph.builder import route_after_knowledge_tree
        assert route_after_knowledge_tree({"stage": "knowledge_tree_done"}) == "chapter_planner"
        assert route_after_knowledge_tree({"stage": "knowledge_tree"}) == "__end__"

    def test_reviewer_routing_reject_retry(self):
        from src.graph.builder import route_after_review
        result = route_after_review({
            "stage": "review_reject", "writer_retry_count": 0,
            "current_post_index": 0, "knowledge_tree": {"topics": ["Topic A"]},
        })
        assert result == "writer_batch"

    def test_reviewer_routing_reject_max_retries(self):
        from src.graph.builder import route_after_review
        from langgraph.graph import END
        result = route_after_review({
            "stage": "review_reject", "writer_retry_count": 3,
            "current_post_index": 0, "knowledge_tree": {"topics": ["Topic A"]},
        })
        assert result == END
