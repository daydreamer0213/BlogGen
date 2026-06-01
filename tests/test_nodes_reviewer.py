"""Test Reviewer nodes: per-chapter review, structure review, review assembly."""
import pytest
from unittest.mock import MagicMock, patch
from src.graph.state import initial_state
from src.agents.nodes import (
    reviewer_single_chapter_node,
    structure_reviewer_node,
    assemble_reviews_node,
    _check_topic_coverage,
    _extract_chapter_draft,
    _affected_chapters,
    _parse_review_markdown,
)


def _mock_llm(content: str, tool_calls=None):
    """Make a mock LLM that always returns the given content."""
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(content=content, tool_calls=tool_calls or [])
    mock.bind_tools.return_value = mock
    mock.model_name = "test-model"
    return mock


PASS_REVIEW = """判断：通过
总评：内容准确，结构清晰
"""

REJECT_REVIEW = """判断：不通过
总评：部分内容需要修正

### 问题1
段落：第2段
类型：事实准确性
严重度：critical
描述：概念定义错误
建议：请使用标准定义
"""

REJECT_PASS_KEYWORD = """判断：通过
总评：好的

### 问题1
段落：第1段
类型：完整性
严重度：minor
描述：缺少示例
建议：补充代码
"""


class TestParseReviewMarkdown:
    def test_parse_pass(self):
        result = _parse_review_markdown(PASS_REVIEW)
        assert result["action"] == "accept"
        assert "内容准确" in result["overall_assessment"]
        assert result["issues"] == []

    def test_parse_reject(self):
        result = _parse_review_markdown(REJECT_REVIEW)
        assert result["action"] == "reject"
        assert len(result["issues"]) == 1
        assert result["issues"][0]["type"] == "事实准确性"
        assert result["issues"][0]["severity"] == "critical"

    def test_parse_pass_but_has_issues__parser_keeps_accept(self):
        """Parser preserves action=accept even when issues are listed under 通过.

        The caller (reviewer_single_chapter_node) overrides action to reject
        if code-based coverage check finds missing topics. The parser itself
        does NOT flip action — it faithfully reports what the LLM wrote.
        """
        result = _parse_review_markdown(REJECT_PASS_KEYWORD)
        # Parser reports what LLM said: "通过"
        assert result["action"] == "accept"
        # But still captures the issues for the caller to handle
        assert len(result["issues"]) == 1
        assert result["issues"][0]["type"] == "完整性"


class TestExtractChapterDraft:
    def test_extract_by_title_match(self):
        assembled = "# Post\n\n## Ch1\n\nCh1 content here.\n\n## Ch2\n\nCh2 content."
        result = _extract_chapter_draft(assembled, "Ch1")
        assert "Ch1 content here." in result
        assert "Ch2" not in result

    def test_extract_last_chapter(self):
        assembled = "# Post\n\n## Ch1\n\nOne\n\n## Ch2\n\nTwo"
        result = _extract_chapter_draft(assembled, "Ch2")
        assert result.strip() == "Two"

    def test_extract_empty_draft(self):
        assert _extract_chapter_draft("", "Ch1") == ""

    def test_extract_not_found(self):
        assert _extract_chapter_draft("## Other\n\ncontent", "Ch1") == ""


class TestCheckTopicCoverage:
    def test_all_covered(self):
        chapter = {"key_points": ["RAG概念", "检索流程"]}
        content = "RAG概念是核心，检索流程也很重要。"
        issues = _check_topic_coverage(chapter, content)
        assert issues == []

    def test_missing_topic(self):
        chapter = {"key_points": ["RAG概念", "检索流程", "与微调对比"]}
        content = "RAG概念是核心，检索流程也很重要。"
        issues = _check_topic_coverage(chapter, content)
        assert len(issues) == 1
        assert "与微调对比" in issues[0]["description"]

    def test_all_missing(self):
        chapter = {"key_points": ["RAG概念"]}
        content = "这篇文章讲的是Agent的内容。"
        issues = _check_topic_coverage(chapter, content)
        assert len(issues) == 1


class TestAffectedChapters:
    def test_specific_chapters(self):
        fb = {"issues": [{"chapter_index": 0}, {"chapter_index": 2}]}
        assert _affected_chapters(fb) == {0, 2}

    def test_duplicate_indices(self):
        fb = {"issues": [{"chapter_index": 1}, {"chapter_index": 1}]}
        assert _affected_chapters(fb) == {1}

    def test_no_chapter_indices_returns_none(self):
        fb = {"issues": [{"description": "no chapter_index"}]}
        assert _affected_chapters(fb) is None

    def test_empty_feedback(self):
        assert _affected_chapters({}) is None


class TestReviewerSingleChapterNode:
    def setup_state(self):
        return {
            **initial_state(),
            "_review_chapter_index": 0,
            "chapter_plan": {
                "post_title": "RAG",
                "chapters": [
                    {"title": "什么是RAG", "key_points": ["RAG概念", "检索流程"]},
                ],
            },
            "assembled_draft": "## 什么是RAG\n\nRAG概念很好理解，检索流程也很简单。",
            "user_needs": {"level": "beginner", "style": "balanced"},
        }

    def test_reviewer_pass(self):
        state = self.setup_state()
        with patch("src.agents.nodes.get_fast_llm", return_value=_mock_llm(PASS_REVIEW)):
            result = reviewer_single_chapter_node(state)
            assert "per_chapter_reviews" in result
            reviews = result["per_chapter_reviews"]
            assert len(reviews) == 1
            assert reviews[0]["chapter_index"] == 0

    def test_reviewer_trusts_llm_judgment(self):
        """Reviewer trusts LLM's judgment — no more code-detected hint injection.

        Substring matching on key_points is unreliable (Writer uses different
        Chinese phrasing). The Reviewer makes its own semantic decision.
        """
        state = self.setup_state()
        state["chapter_plan"]["chapters"][0]["key_points"].append("NOT_IN_CONTENT")
        with patch("src.agents.nodes.get_fast_llm", return_value=_mock_llm(PASS_REVIEW)):
            result = reviewer_single_chapter_node(state)
            review = result["per_chapter_reviews"][0]["review"]
            # LLM says pass → accept. No code-detected hints injected.
            assert review["action"] == "accept"
            assert len(review["issues"]) == 0

    def test_reviewer_no_content_found(self):
        """When chapter draft can't be extracted from assembled."""
        state = self.setup_state()
        state["assembled_draft"] = ""
        with patch("src.agents.nodes.get_fast_llm", return_value=_mock_llm(PASS_REVIEW)):
            result = reviewer_single_chapter_node(state)
            assert len(result["per_chapter_reviews"]) == 1

    def test_reviewer_chapter_index_out_of_range(self):
        state = self.setup_state()
        state["_review_chapter_index"] = 5
        result = reviewer_single_chapter_node(state)
        assert result["per_chapter_reviews"] == []


class TestAssembleReviews:
    def test_all_pass(self):
        state = {
            **initial_state(),
            "per_chapter_reviews": [
                {"chapter_index": 0, "review": {"action": "accept", "word_count": 500, "issues": []}},
                {"chapter_index": 1, "review": {"action": "accept", "word_count": 600, "issues": []}},
            ],
            "structure_review": {"action": "accept", "issues": []},
            "assembled_draft": "# RAG\n\n## Ch1\n\ncontent\n\n## Ch2\n\ncontent",
        }
        result = assemble_reviews_node(state)
        assert result["stage"] == "review_pass"
        assert result["review_result"]["action"] == "accept"
        assert "final" in result

    def test_one_chapter_rejected(self):
        state = {
            **initial_state(),
            "per_chapter_reviews": [
                {"chapter_index": 0, "review": {"action": "accept", "word_count": 500, "issues": []}},
                {"chapter_index": 1, "review": {
                    "action": "reject", "word_count": 300,
                    "issues": [{"description": "内容缺失", "severity": "critical"}],
                }},
            ],
            "structure_review": {"action": "accept", "issues": []},
            "assembled_draft": "content",
        }
        result = assemble_reviews_node(state)
        assert result["stage"] == "review_reject"
        assert result["review_result"]["action"] == "reject"
        assert len(result["review_feedback"]["issues"]) == 1

    def test_structure_review_rejected(self):
        state = {
            **initial_state(),
            "per_chapter_reviews": [
                {"chapter_index": 0, "review": {"action": "accept", "word_count": 500, "issues": []}},
            ],
            "structure_review": {
                "action": "reject",
                "issues": [{"description": "章节顺序不合理", "severity": "critical"}],
            },
            "assembled_draft": "content",
        }
        result = assemble_reviews_node(state)
        assert result["stage"] == "review_reject"
        assert len(result["review_feedback"]["issues"]) == 1

    def test_no_reviews_errors(self):
        state = {**initial_state(), "per_chapter_reviews": [], "structure_review": {}}
        result = assemble_reviews_node(state)
        assert "_error" in result

    def test_clears_accumulator_arrays(self):
        state = {
            **initial_state(),
            "per_chapter_reviews": [
                {"chapter_index": 0, "review": {"action": "accept", "word_count": 500, "issues": []}},
            ],
            "structure_review": {"action": "accept", "issues": []},
            "assembled_draft": "draft",
        }
        result = assemble_reviews_node(state)
        assert result["per_chapter_reviews"] == []
        assert result["structure_review"] == {}


class TestStructureReviewerNode:
    def test_structure_reviewer_runs(self):
        state = {
            **initial_state(),
            "chapter_plan": {
                "post_title": "RAG",
                "chapters": [
                    {"title": "Ch1", "key_points": ["kp1", "kp2"]},
                ],
            },
            "assembled_draft": "## Ch1\n\nSome content here.",
            "user_needs": {"level": "beginner", "style": "balanced"},
        }
        review_output = "判断：通过\n总评：结构合理，内容完整。\n"
        with patch("src.agents.nodes.get_fast_llm", return_value=_mock_llm(review_output)):
            result = structure_reviewer_node(state)
            assert "structure_review" in result
            assert result["structure_review"]["action"] == "accept"


class TestReviewerAssemblerPipeline:
    """Reviewer → AssembleReviews → state for retry or pass."""

    def test_full_review_flow_pass(self):
        draft = "## 什么是RAG\n\nRAG概念的核心是检索增强生成，通过知识库检索来增强大模型能力。\n\n## 检索流程\n\n检索流程很重要，包括索引构建和相似度搜索。"
        state = {
            **initial_state(),
            "chapter_plan": {
                "post_title": "RAG入门",
                "chapters": [
                    {"title": "什么是RAG", "key_points": ["RAG概念"]},
                    {"title": "检索流程", "key_points": ["检索流程"]},
                ],
            },
            "assembled_draft": draft,
            "user_needs": {"level": "beginner", "style": "balanced"},
        }

        # Chapter 0 review
        with patch("src.agents.nodes.get_fast_llm", return_value=_mock_llm(PASS_REVIEW)):
            r0 = reviewer_single_chapter_node({**state, "_review_chapter_index": 0})
        # Chapter 1 review
        with patch("src.agents.nodes.get_fast_llm", return_value=_mock_llm(PASS_REVIEW)):
            r1 = reviewer_single_chapter_node({**state, "_review_chapter_index": 1})
        # Structure review (uses get_fast_llm after Flash migration)
        with patch("src.agents.nodes.get_fast_llm", return_value=_mock_llm("判断：通过\n总评：好\n")):
            sr = structure_reviewer_node(state)

        merged_state = {**state, **r0, **r1, **sr}
        final = assemble_reviews_node(merged_state)
        assert final["stage"] == "review_pass"
        assert final["final"] == draft

    def test_full_review_flow_reject(self):
        """One chapter rejected → merged result is reject."""
        draft = "## 什么是RAG\n\nRAG概念。\n\n## 检索流程\n\n检索流程。"
        state = {
            **initial_state(),
            "chapter_plan": {
                "post_title": "RAG入门",
                "chapters": [
                    {"title": "什么是RAG", "key_points": ["RAG概念"]},
                    {"title": "检索流程", "key_points": ["检索流程"]},
                ],
            },
            "assembled_draft": draft,
            "user_needs": {"level": "beginner", "style": "balanced"},
        }

        with patch("src.agents.nodes.get_fast_llm", return_value=_mock_llm(PASS_REVIEW)):
            r0 = reviewer_single_chapter_node({**state, "_review_chapter_index": 0})
        with patch("src.agents.nodes.get_fast_llm", return_value=_mock_llm(REJECT_REVIEW)):
            r1 = reviewer_single_chapter_node({**state, "_review_chapter_index": 1})
        with patch("src.agents.nodes.get_fast_llm", return_value=_mock_llm("判断：通过\n总评：好\n")):
            sr = structure_reviewer_node(state)

        merged_state = {**state, **r0, **r1, **sr}
        final = assemble_reviews_node(merged_state)
        assert final["stage"] == "review_reject"
        assert len(final["review_feedback"]["issues"]) >= 1


class TestTier1CheckNode:
    """Tier1 code-level check: no LLM, pure Python rules."""

    def test_all_pass(self):
        from src.agents.nodes import tier1_check_node
        content = "RAG概念是检索增强生成的核心思想。" * 20  # >300 chars for min check
        state = {
            "chapter_plan": {
                "chapters": [
                    {"title": "Ch1", "key_points": ["RAG概念"]},
                ],
            },
            "per_chapter_drafts": [
                {"chapter_index": 0, "chapter_title": "Ch1", "draft_content": content},
            ],
            "user_needs": {"level": "beginner"},
        }
        result = tier1_check_node(state)
        assert result["tier1_pass"] is True

    def test_missing_topic_no_longer_blocked_by_tier1(self):
        """Tier1 no longer checks topic coverage (substring FP too high).
        Coverage is now Reviewer's responsibility."""
        from src.agents.nodes import tier1_check_node
        content = "RAG概念讲得很清楚。" * 30  # >300 chars for min check
        state = {
            "chapter_plan": {
                "chapters": [
                    {"title": "Ch1", "key_points": ["RAG概念", "缺失知识点"]},
                ],
            },
            "per_chapter_drafts": [
                {"chapter_index": 0, "chapter_title": "Ch1", "draft_content": content},
            ],
            "user_needs": {"level": "beginner"},
        }
        result = tier1_check_node(state)
        assert result["tier1_pass"] is True

    def test_code_block_too_long_is_minor(self):
        """Code block >30 lines is minor, does not block (Reviewer handles)."""
        from src.agents.nodes import tier1_check_node
        base = "RAG概念是检索增强生成的核心思想，通过外部知识库检索来增强大模型的回答质量。" * 3
        content = base + "\n\n```python\n" + "\n".join(f"line_{i}" for i in range(35)) + "\n```"
        state = {
            "chapter_plan": {
                "chapters": [
                    {"title": "Ch1", "key_points": ["RAG概念"]},
                ],
            },
            "per_chapter_drafts": [
                {"chapter_index": 0, "chapter_title": "Ch1", "draft_content": content},
            ],
            "user_needs": {"level": "beginner"},
        }
        result = tier1_check_node(state)
        assert result["tier1_pass"] is True  # minor, does not block

    def test_empty_chapter(self):
        from src.agents.nodes import tier1_check_node
        state = {
            "chapter_plan": {
                "chapters": [
                    {"title": "不存在", "key_points": []},
                ],
            },
            "per_chapter_drafts": [],
            "user_needs": {"level": "beginner"},
        }
        result = tier1_check_node(state)
        assert result["tier1_pass"] is False

    def test_very_short_chapter(self):
        """<50 chars = title shell, should trigger retry."""
        from src.agents.nodes import tier1_check_node
        state = {
            "chapter_plan": {
                "chapters": [
                    {"title": "Ch1", "key_points": ["RAG概念"]},
                ],
            },
            "per_chapter_drafts": [
                {"chapter_index": 0, "chapter_title": "Ch1", "draft_content": "这是很短的内容。"},
            ],
            "user_needs": {"level": "beginner"},
        }
        result = tier1_check_node(state)
        assert result["tier1_pass"] is False
        assert any("低于50字" in i["description"] for i in result["review_feedback"]["issues"])
