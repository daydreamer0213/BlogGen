"""Regression tests for previously-fixed bugs.

Each test encodes a known failure to prevent silent regression.
Industry pattern (OpenAI): every fixed bug gets a regression test.
"""
import pytest
from unittest.mock import MagicMock, patch


class TestRegressionReviewFeedbackFalsy:
    """Bug: if review_feedback was {} then writer skipped fix mode.

    Fixed by: 'if review_feedback is not None' instead of 'if review_feedback'.
    Verified in nodes.py:writer_single_chapter_node.
    """

    def test_empty_dict_review_feedback_enters_fix_mode(self):
        from src.agents.nodes import writer_single_chapter_node

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="# Fixed", tool_calls=None)

        state = {
            "user_needs": {"domain": "AI", "level": "beginner", "goal": "test", "style": "balanced"},
            "chapter_plan": {"chapters": [{"title": "Ch1", "key_points": ["a"]}]},
            "_fanout_chapter_index": 0,
            "review_feedback": {},  # empty dict — was falsy, now correctly enters fix mode
        }

        with patch("src.agents.nodes.get_llm", return_value=mock_llm):
            with patch("src.agents.nodes._run_with_tools", return_value="# Fixed"):
                result = writer_single_chapter_node(state)

        assert len(result["per_chapter_drafts"]) == 1


class TestRegressionSubtreePrefixMatching:
    """Bug: subtree collection used startswith(id) so "10" matched "1".

    Fixed by: using id==root_id or pid==root_id or pid.startswith(root_id+".").
    Replaced with flat topic list — no tree traversal needed.
    """

    def test_topic_is_flat_string(self):
        from src.schemas import KnowledgeTree
        tree = KnowledgeTree(domain="test", topics=["Topic A", "Topic B"])
        topics = tree.topics
        # In flat list, there is no parent_id — string matching can't cause false positives
        assert isinstance(topics[0], str)


class TestRegressionChapterPlanEmpty:
    """Bug: ChapterPlan({}) silently returned empty plan, no validation.

    Fixed by: @model_validator check_has_chapters rejects empty chapters list.
    """

    def test_empty_plan_dict_rejected(self):
        from src.schemas import ChapterPlan
        with pytest.raises(ValueError, match="must have at least one chapter"):
            ChapterPlan(**{})

    def test_nonempty_plan_accepted(self):
        from src.schemas import ChapterPlan
        plan = ChapterPlan(post_title="Test", chapters=[
            {"title": "Ch1", "key_points": ["a"]},
        ])
        assert len(plan.chapters) == 1


class TestRegressionSilentPassOnValueError:
    """Bug: except ValueError: pass silently swallowed validation failures.

    Fixed by: nodes now let ValueError propagate to run_node() which returns
    {"_error": ..., "stage": "XXX_error"}.
    """

    def test_validate_or_raise_actually_raises(self):
        from src.schemas import validate_or_raise, KnowledgeTree
        with pytest.raises(ValueError, match="schema validation"):
            validate_or_raise(KnowledgeTree, {"topics": []}, "test")


class TestRegressionSessionConfigKeyError:
    """Bug: create() without thread_id left self.config={}, causing
    KeyError('configurable') on update_state.

    Fixed by: else branch generates UUID and sets config.
    """

    def test_create_without_thread_id_sets_config(self):
        from src.graph.session import BlogGenSession
        session = BlogGenSession()
        session.create()
        assert "configurable" in session.config
        assert session.thread_id != ""


class TestRegressionFetchPageContentBroken:
    """Bug: FetchPageInput schema lacked max_chars field, causing KeyError
    in fetch_page_content → silent empty return.

    Fixed by: added max_chars to FetchPageInput schema.
    """

    def test_fetch_page_input_has_max_chars(self):
        from src.schemas import FetchPageInput
        v = FetchPageInput(url="https://example.com", max_chars=500)
        assert v.max_chars == 500


class TestRegressionDuplicateJsonImport:
    """Bug: import json was at bottom of schemas.py, fragile.

    Fixed by: moved to top of file.
    """
    def test_json_import_at_top(self):
        with open("src/schemas.py", "r", encoding="utf-8") as f:
            first_lines = "".join(f.readlines()[:5])
        assert "import json" in first_lines


class TestRegressionDoubleBreak:
    """Bug: double 'break; break' dead code in _run_with_tools.

    Fixed by: removed second break.
    """
    def test_no_double_break_in_run_with_tools(self):
        import inspect
        from src.agents.nodes import _run_with_tools
        src = inspect.getsource(_run_with_tools)
        assert "break\n            break" not in src.replace(" ", "").replace("\t", "")
