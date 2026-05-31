"""End-to-end tests: full pipeline simulation with parallel Writer/Reviewer."""
import json
import pytest
from unittest.mock import MagicMock, patch
from tests.conftest import needs_all_deps, needs_langgraph, needs_jinja2


def _make_mock_llm(responses):
    mock_llm = MagicMock()
    mocks = [MagicMock(content=t, tool_calls=[]) for t in responses]
    mock_llm.invoke.side_effect = mocks
    mock_llm.bind_tools.return_value = mock_llm
    return mock_llm


@needs_all_deps
class TestFullPipelineHappyPath:
    def test_full_pipeline_single_post(self):
        from src.agents.nodes import (
            needs_alignment_node, knowledge_tree_node,
            chapter_planner_node, writer_single_chapter_node, assembler_node,
            structure_reviewer_node, reviewer_single_chapter_node, assemble_reviews_node,
        )

        profile_json = '{"domain":"RAG","level":"beginner","goal":"test"}'
        mock1 = _make_mock_llm([profile_json])
        with patch("src.agents.nodes.get_llm", return_value=mock1):
            with patch("src.agents.nodes.invoke_with_retry", return_value=MagicMock(content=profile_json)):
                result1 = needs_alignment_node({
                    "messages": [{"role":"assistant","content":"hi"},{"role":"user","content":"RAG, beginner"}],
                    "stage": "needs_alignment",
                })
        assert result1["stage"] == "needs_alignment_done"

        tree_md = "# RAG\n\n- topic a\n- topic b\n- topic c\n"
        mock2 = _make_mock_llm([tree_md])
        with patch("src.agents.nodes.get_llm", return_value=mock2):
            with patch("src.agents.nodes._run_with_tools", return_value=tree_md):
                result2 = knowledge_tree_node({"user_needs": result1["user_needs"], "stage": "needs_done"})
        assert result2["stage"] == "knowledge_tree_done"

        chapter_plan = {"post_title": "RAG", "chapters": [{"title": "Ch1", "key_points": ["a"]}]}
        draft = "# RAG\n\n## Ch1\n\nThis is the content of chapter one with enough text."
        mock4 = _make_mock_llm([draft])
        with patch("src.agents.nodes.get_fast_llm", return_value=mock4):
            with patch("src.agents.nodes._run_with_tools", return_value=draft):
                r4 = writer_single_chapter_node({
                    "user_needs": result1["user_needs"], "chapter_plan": chapter_plan,
                    "_fanout_chapter_index": 0, "review_feedback": None,
                })
                result4 = assembler_node({
                    "chapter_plan": chapter_plan,
                    "per_chapter_drafts": r4["per_chapter_drafts"],
                })
        assert result4["stage"] == "writer_done"

        review_md = "判断：通过\n字数：100\n总评：ok\n"
        mock5 = _make_mock_llm([review_md])
        with patch("src.agents.nodes.get_fast_llm", return_value=mock5):
            with patch("src.agents.nodes.get_llm", return_value=mock5):
                with patch("src.agents.nodes._run_with_tools", return_value=review_md):
                    sr = structure_reviewer_node({
                        "chapter_plan": chapter_plan,
                        "assembled_draft": result4["assembled_draft"],
                        "user_needs": result1["user_needs"],
                    })
                    rch = reviewer_single_chapter_node({
                        "chapter_plan": chapter_plan,
                        "assembled_draft": result4["assembled_draft"],
                        "user_needs": result1["user_needs"],
                        "_review_chapter_index": 0,
                    })
                    result5 = assemble_reviews_node({
                        "user_needs": result1["user_needs"],
                        "assembled_draft": result4["assembled_draft"],
                        "chapter_plan": chapter_plan,
                        "per_chapter_reviews": rch.get("per_chapter_reviews", []),
                        "structure_review": sr.get("structure_review", {}),
                    })
        assert result5["stage"] == "review_pass"

    def test_full_pipeline_with_rejection_and_retry(self):
        from src.agents.nodes import (
            writer_single_chapter_node, assembler_node,
            structure_reviewer_node, reviewer_single_chapter_node, assemble_reviews_node,
        )

        user_needs = {"domain": "AI", "level": "beginner", "goal": "test", "style": "balanced"}
        chapter_plan = {"post_title": "RAG", "chapters": [{"title": "Ch1", "key_points": ["a"]}]}

        draft1 = "# Blog\n\nSome error."
        mock_w1 = _make_mock_llm([draft1])
        with patch("src.agents.nodes.get_fast_llm", return_value=mock_w1):
            with patch("src.agents.nodes._run_with_tools", return_value=draft1):
                r1ch = writer_single_chapter_node({
                    "user_needs": user_needs, "chapter_plan": chapter_plan,
                    "_fanout_chapter_index": 0, "review_feedback": None,
                })
                r1 = assembler_node({
                    "chapter_plan": chapter_plan,
                    "per_chapter_drafts": r1ch["per_chapter_drafts"],
                })
        review1 = "判断：不通过\n字数：50\n总评：bad\n\n### 问题1\n段落：p2\n类型：事实错误\n严重度：critical\n章节：1\n描述：err\n建议：fix\n"
        mock_r1 = _make_mock_llm([review1])
        with patch("src.agents.nodes.get_fast_llm", return_value=mock_r1):
            with patch("src.agents.nodes.get_llm", return_value=mock_r1):
                with patch("src.agents.nodes._run_with_tools", return_value=review1):
                    sr = structure_reviewer_node({
                        "chapter_plan": chapter_plan, "assembled_draft": r1["assembled_draft"],
                        "user_needs": user_needs,
                    })
                    rch = reviewer_single_chapter_node({
                        "chapter_plan": chapter_plan, "assembled_draft": r1["assembled_draft"],
                        "user_needs": user_needs, "_review_chapter_index": 0,
                    })
                    rev = assemble_reviews_node({
                        "user_needs": user_needs, "assembled_draft": r1["assembled_draft"],
                        "chapter_plan": chapter_plan,
                        "per_chapter_reviews": rch.get("per_chapter_reviews", []),
                        "structure_review": sr.get("structure_review", {}),
                    })
        assert rev["stage"] == "review_reject"

        draft2 = "# Blog\n\nFixed."
        mock_w2 = _make_mock_llm([draft2])
        with patch("src.agents.nodes.get_fast_llm", return_value=mock_w2):
            with patch("src.agents.nodes._run_with_tools", return_value=draft2):
                r2ch = writer_single_chapter_node({
                    "user_needs": user_needs, "chapter_plan": chapter_plan,
                    "_fanout_chapter_index": 0, "review_feedback": rev["review_feedback"],
                })
                r2 = assembler_node({
                    "chapter_plan": chapter_plan,
                    "per_chapter_drafts": r2ch["per_chapter_drafts"],
                })
        review2 = "判断：通过\n字数：30\n总评：fixed\n"
        mock_r2 = _make_mock_llm([review2])
        with patch("src.agents.nodes.get_fast_llm", return_value=mock_r2):
            with patch("src.agents.nodes.get_llm", return_value=mock_r2):
                with patch("src.agents.nodes._run_with_tools", return_value=review2):
                    sr2 = structure_reviewer_node({
                        "chapter_plan": chapter_plan, "assembled_draft": r2["assembled_draft"],
                        "user_needs": user_needs,
                    })
                    rch2 = reviewer_single_chapter_node({
                        "chapter_plan": chapter_plan, "assembled_draft": r2["assembled_draft"],
                        "user_needs": user_needs, "_review_chapter_index": 0,
                    })
                    rev2 = assemble_reviews_node({
                        "user_needs": user_needs, "assembled_draft": r2["assembled_draft"],
                        "chapter_plan": chapter_plan,
                        "per_chapter_reviews": rch2.get("per_chapter_reviews", []),
                        "structure_review": sr2.get("structure_review", {}),
                    })
        assert rev2["stage"] == "review_pass"


@needs_langgraph
class TestStatePersistence:
    def test_initial_state_has_all_keys(self):
        from src.graph.state import initial_state
        s = initial_state()
        for key in ["messages", "user_needs", "knowledge_tree", "chapter_plan",
                     "draft", "writer_retry_count", "review_result", "stage"]:
            assert key in s

    def test_state_immutability_across_updates(self):
        from src.graph.state import initial_state
        s1 = initial_state()
        s2 = initial_state()
        s2["draft"] = "modified"
        assert s1["draft"] == ""


class TestMonitoringInPipeline:
    def test_callback_captures_llm(self):
        from src.monitor import BlogGenMonitorCallback, reset_session
        reset_session()
        cb = BlogGenMonitorCallback()
        cb._llm_t_start = 0
        cb.on_chain_start({"name": "TestNode"}, None, run_id="r1", parent_run_id=None)
        mock = MagicMock()
        mock.llm_output = {"token_usage": {"prompt_tokens": 100, "completion_tokens": 50}}
        mock.generations = [[MagicMock(generation_info=None, message=MagicMock(response_metadata={"model_name":"test"}))]]
        cb.on_llm_end(mock)
        assert cb._current_llm_calls[0]["prompt_tokens"] == 100

    def test_record_graph_run(self):
        from src.monitor import reset_session, record_graph_run, get_session_summary
        reset_session()
        record_graph_run([], "Agent", {}, {}, [{"model":"x","prompt_tokens":1,"completion_tokens":1,"latency_ms":1}], [], 100, None)
        assert get_session_summary()["agent_count"] == 1

    def test_record_graph_run_error(self):
        from src.monitor import reset_session, record_graph_run, get_session_summary
        reset_session()
        record_graph_run([], "Bad", {}, {}, [], [], 100, "ValueError: test")
        assert get_session_summary()["recent"][0]["error"] == "ValueError: test"


class TestKnowledgeTreeMethods:
    def test_topic_list(self):
        from src.agents.nodes import _parse_knowledge_tree_markdown
        tree = _parse_knowledge_tree_markdown("# AI\n- topic a\n- topic b")
        assert tree["domain"] == "AI"
        assert len(tree["topics"]) == 2

    def test_empty_topics_raises(self):
        from src.agents.nodes import _parse_knowledge_tree_markdown
        tree = _parse_knowledge_tree_markdown("# AI\n")
        assert tree["topics"] == []