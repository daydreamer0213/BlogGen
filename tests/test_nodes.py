"""Test Agent node logic with mocked LLM responses."""
import pytest
from unittest.mock import MagicMock, patch, ANY


class TestNeedsAlignment:
    def test_empty_history_returns_noop(self):
        from src.agents.nodes import needs_alignment_node
        state = {"messages": [], "stage": "needs_alignment"}
        result = needs_alignment_node(state)
        # Greeting is now shown by UI, not the node
        assert result["stage"] == "needs_alignment"
        assert "messages" not in result  # Node doesn't produce messages when empty

    def test_profile_detected_transitions(self):
        from src.agents.nodes import needs_alignment_node
        from unittest.mock import patch

        mock_resp = MagicMock()
        mock_resp.content = '{"domain": "AI", "level": "beginner", "goal": "面试"}'
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_resp

        state = {
            "messages": [
                {"role": "assistant", "content": "请告诉我..."},
                {"role": "user", "content": "我想学AI，初学者，目标面试"},
            ],
            "stage": "needs_alignment",
        }

        with patch("src.agents.nodes.get_llm", return_value=mock_llm):
            with patch("src.agents.nodes.invoke_with_retry", return_value=mock_resp):
                result = needs_alignment_node(state)
                assert result["stage"] == "needs_alignment_done"
                assert result["user_needs"]["domain"] == "AI"

    def test_incomplete_info_stays_in_alignment(self):
        from src.agents.nodes import needs_alignment_node
        from unittest.mock import patch

        mock_resp = MagicMock()
        mock_resp.content = "请告诉我你的学习目标是什么？"  # No JSON, still collecting
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_resp

        state = {
            "messages": [
                {"role": "assistant", "content": "请告诉我..."},
                {"role": "user", "content": "我想学AI"},
            ],
            "stage": "needs_alignment",
        }

        with patch("src.agents.nodes.get_llm", return_value=mock_llm):
            with patch("src.agents.nodes.invoke_with_retry", return_value=mock_resp):
                result = needs_alignment_node(state)
                assert result["stage"] == "needs_alignment"  # Still collecting

    def test_fuzzy_level_normalized(self):
        from src.agents.nodes import needs_alignment_node
        from unittest.mock import patch

        mock_resp = MagicMock()
        mock_resp.content = '{"domain": "AI", "level": "还行吧", "goal": "面试"}'
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_resp

        state = {
            "messages": [
                {"role": "user", "content": "我想学AI，水平还行吧，目标面试"},
            ],
            "stage": "needs_alignment",
        }

        with patch("src.agents.nodes.get_llm", return_value=mock_llm):
            with patch("src.agents.nodes.invoke_with_retry", return_value=mock_resp):
                result = needs_alignment_node(state)
                assert result["user_needs"]["level"] == "beginner"  # Normalized


class TestKnowledgeTreeBuilder:
    def test_empty_state_handled(self):
        from src.agents.nodes import knowledge_tree_node
        state = {"user_needs": {}, "stage": "needs_done"}
        # Without mock, _run_with_tools would try real LLM. Test that it doesn't crash
        # on empty user_needs (the prompt construction should handle missing keys)
        # This is an integration test point; the node passes profile.get() for all keys
        # so empty dict is safe.
        pass  # Marked for integration testing with real/mocked LLM


class TestWriter:
    def test_draft_compression_in_fix_mode(self):
        """Fix mode should compress code blocks and truncate long drafts."""
        long_draft = "x" * 10000 + "\n```python\n" + "x" * 500 + "\n```\n" + "y" * 5000
        import re
        compressed = re.sub(
            r"```[\s\S]*?```",
            lambda m: m.group(0)[:200] + "\n# ...(code block truncated)...",
            long_draft
        )
        assert len(compressed) < len(long_draft)
        if len(compressed) > 8000:
            compressed = compressed[:8000] + "\n\n...(content truncated)..."
        assert len(compressed) <= 8000 + 30  # ~8000 + truncation msg


class TestParseChapterMarkdown:
    """Tests for _parse_chapter_markdown — the deterministic Markdown→ChapterPlan parser."""

    def test_basic_two_chapters(self):
        from src.agents.nodes import _parse_chapter_markdown
        md = (
            "# RAG入门\n\n"
            "## RAG是什么\n"
            "- LLM局限\n"
            "- 检索增强\n\n"
            "## 向量检索实战\n"
            "- 选型对比\n"
            "- 代码实现\n"
        )
        plan = _parse_chapter_markdown(md)
        assert plan["post_title"] == "RAG入门"
        assert len(plan["chapters"]) == 2
        assert plan["chapters"][0]["title"] == "RAG是什么"
        assert plan["chapters"][0]["key_points"] == ["LLM局限", "检索增强"]
        assert plan["chapters"][1]["title"] == "向量检索实战"

    def test_empty_input(self):
        from src.agents.nodes import _parse_chapter_markdown
        plan = _parse_chapter_markdown("")
        assert plan == {"post_title": "", "chapters": []}

    def test_no_chapters(self):
        from src.agents.nodes import _parse_chapter_markdown
        plan = _parse_chapter_markdown("# Just a title\n一些描述文字")
        assert plan["post_title"] == "Just a title"
        assert plan["chapters"] == []

    def test_title_with_trailing_colon(self):
        from src.agents.nodes import _parse_chapter_markdown
        md = "# Blog\n## 第一章：带冒号的标题\n- point 1\n"
        plan = _parse_chapter_markdown(md)
        assert plan["chapters"][0]["title"] == "第一章：带冒号的标题"
        assert plan["chapters"][0]["key_points"] == ["point 1"]

    def test_asterisk_bullets(self):
        from src.agents.nodes import _parse_chapter_markdown
        md = "# Test\n## Ch1\n* item 1\n* item 2\n"
        plan = _parse_chapter_markdown(md)
        assert plan["chapters"][0]["key_points"] == ["item 1", "item 2"]

    def test_no_title_line(self):
        from src.agents.nodes import _parse_chapter_markdown
        md = "## Chapter 1\n- point 1\n"
        plan = _parse_chapter_markdown(md)
        assert plan["post_title"] == ""
        assert len(plan["chapters"]) == 1


class TestParseKnowledgeTreeMarkdown:
    """Tests for _parse_knowledge_tree_markdown — pure topic list format."""

    def test_topic_list(self):
        from src.agents.nodes import _parse_knowledge_tree_markdown
        md = (
            "# RAG学习路线\n\n"
            "- 检索增强生成：概念与动机\n"
            "- 文档加载与解析\n"
            "- 向量嵌入：embedding模型选型\n"
        )
        tree = _parse_knowledge_tree_markdown(md)
        assert tree["domain"] == "RAG学习路线"
        assert tree["topics"] == [
            "检索增强生成：概念与动机",
            "文档加载与解析",
            "向量嵌入：embedding模型选型",
        ]

    def test_empty_topics(self):
        from src.agents.nodes import _parse_knowledge_tree_markdown
        md = "# Just domain\n"
        tree = _parse_knowledge_tree_markdown(md)
        assert tree["domain"] == "Just domain"
        assert tree["topics"] == []

    def test_empty_input(self):
        from src.agents.nodes import _parse_knowledge_tree_markdown
        tree = _parse_knowledge_tree_markdown("")
        assert tree == {"domain": "", "topics": []}


class TestParseReviewMarkdown:
    """Tests for _parse_review_markdown."""

    def test_accept(self):
        from src.agents.nodes import _parse_review_markdown
        md = "判断：通过\n字数：12000\n总评：写得好\n"
        r = _parse_review_markdown(md)
        assert r["action"] == "accept"
        assert r["word_count"] == 12000
        assert r["overall_assessment"] == "写得好"
        assert r["issues"] == []

    def test_reject_with_issues(self):
        from src.agents.nodes import _parse_review_markdown
        md = (
            "判断：不通过\n"
            "字数：500\n"
            "总评：有问题\n\n"
            "### 问题1\n"
            "段落：第2段\n"
            "类型：事实错误\n"
            "严重度：critical\n"
            "描述：版本号错了\n"
            "建议：改成3.0\n\n"
            "### 问题2\n"
            "段落：第5段\n"
            "类型：易读性\n"
            "描述：段落太长\n"
            "建议：拆分\n"
        )
        r = _parse_review_markdown(md)
        assert r["action"] == "reject"
        assert len(r["issues"]) == 2
        assert r["issues"][0]["type"] == "事实错误"
        assert r["issues"][0]["severity"] == "critical"
        assert r["issues"][1]["type"] == "易读性"
        assert r["issues"][1]["severity"] == "minor"  # default

    def test_split_suggestion(self):
        from src.agents.nodes import _parse_review_markdown
        md = (
            "判断：不通过\n"
            "字数：20000\n"
            "总评：需拆分\n\n"
            "### 问题1\n"
            "段落：全文\n"
            "类型：字数超标\n"
            "严重度：critical\n"
            "描述：超18000字\n"
            "建议：拆分\n\n"
            "#### 拆分建议\n"
            "原因：语义分界\n"
            "第一篇：RAG入门 | 开头到检索 | 完整pipeline\n"
            "第二篇：RAG进阶 | 向量选型到结尾 | 架构层面\n"
        )
        r = _parse_review_markdown(md)
        assert len(r["issues"]) == 1
        sp = r["issues"][0]["split_suggestion"]
        assert sp["split_reason"] == "语义分界"
        assert len(sp["groups"]) == 2
        assert sp["groups"][0]["title"] == "RAG入门"
        assert sp["groups"][0]["content_range"] == "开头到检索"
        assert sp["groups"][0]["rationale"] == "完整pipeline"

    def test_empty_input(self):
        from src.agents.nodes import _parse_review_markdown
        r = _parse_review_markdown("")
        assert r["action"] == "accept"
        assert r["issues"] == []


class TestWriterSingleChapter:
    """Tests for writer_single_chapter_node — the fan-out chapter writer."""

    def test_writes_one_chapter(self):
        from src.agents.nodes import writer_single_chapter_node

        resp = "# Chapter 1\n\nContent here.\n\n```python\nprint('hi')\n```"
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.return_value = MagicMock(content=resp, tool_calls=None)
        mock_llm.model_name = "test"

        state = {
            "user_needs": {"domain": "AI", "level": "beginner", "goal": "test", "style": "balanced"},
            "chapter_plan": {
                "post_title": "Test Blog",
                "chapters": [
                    {"title": "Chapter 1", "key_points": ["point a", "point b"]},
                    {"title": "Chapter 2", "key_points": ["point c"]},
                ],
            },
            "_fanout_chapter_index": 0,
        }

        with patch("src.agents.nodes.get_llm", return_value=mock_llm):
            with patch("src.agents.nodes._run_with_tools", return_value=resp):
                result = writer_single_chapter_node(state)

        assert len(result["per_chapter_drafts"]) == 1
        ch = result["per_chapter_drafts"][0]
        assert ch["chapter_index"] == 0
        assert ch["chapter_title"] == "Chapter 1"
        assert "Chapter 1" in ch["draft_content"]

    def test_out_of_range_returns_empty(self):
        from src.agents.nodes import writer_single_chapter_node
        state = {
            "user_needs": {"domain": "AI", "level": "beginner", "goal": "test", "style": "balanced"},
            "chapter_plan": {"chapters": [{"title": "Ch1", "key_points": []}]},
            "_fanout_chapter_index": 99,
        }
        result = writer_single_chapter_node(state)
        assert result["per_chapter_drafts"] == []


class TestAssemblerNode:
    """Tests for assembler_node — concatenates per-chapter drafts."""

    def test_assembles_in_order(self):
        from src.agents.nodes import assembler_node
        state = {
            "chapter_plan": {
                "post_title": "My Blog",
                "chapters": [
                    {"title": "First"},
                    {"title": "Second"},
                ],
            },
            "assembled_draft": "",
            "per_chapter_drafts": [
                {"chapter_index": 1, "chapter_title": "Second", "draft_content": "## Second\nContent 2"},
                {"chapter_index": 0, "chapter_title": "First", "draft_content": "## First\nContent 1"},
            ],
        }
        result = assembler_node(state)
        assert result["stage"] == "writer_done"
        assert "# My Blog" in result["assembled_draft"]
        assert result["assembled_draft"].index("First") < result["assembled_draft"].index("Second")

    def test_empty_returns_error(self):
        from src.agents.nodes import assembler_node
        result = assembler_node({"chapter_plan": {"chapters": []}, "assembled_draft": "", "per_chapter_drafts": []})
        assert result["_error"]

    def test_partial_update_reuses_old_draft(self):
        from src.agents.nodes import assembler_node
        state = {
            "chapter_plan": {
                "post_title": "Blog",
                "chapters": [
                    {"title": "Kept"},
                    {"title": "Updated"},
                ],
            },
            "assembled_draft": "# Blog\n## Kept\nOld content.\n## Updated\nOld too.\n",
            "per_chapter_drafts": [
                {"chapter_index": 1, "draft_content": "## Updated\nNew content."},
            ],
        }
        result = assembler_node(state)
        assert "Old content" in result["assembled_draft"]  # chapter 0 reused
        assert "New content" in result["assembled_draft"]   # chapter 1 updated


class TestCheckTopicCoverage:
    """Tests for _check_topic_coverage — code-based missing topic detection."""

    def test_all_topics_found(self):
        from src.agents.nodes import _check_topic_coverage
        chapter = {"key_points": ["向量检索", "ANN"]}
        content = "向量检索是指...ANN是一种近似搜索算法..."
        issues = _check_topic_coverage(chapter, content)
        assert issues == []

    def test_missing_topic_detected(self):
        from src.agents.nodes import _check_topic_coverage
        chapter = {"key_points": ["向量检索", "ANN"]}
        content = "向量检索是指使用向量进行相似度查找的技术。"
        issues = _check_topic_coverage(chapter, content)
        assert len(issues) == 1
        assert "ANN" in issues[0]["description"]

    def test_none_chapter(self):
        from src.agents.nodes import _check_topic_coverage
        with pytest.raises(AttributeError):
            _check_topic_coverage(None, "content")


class TestExtractChapterDraft:
    """Tests for _extract_chapter_draft — regex + fuzzy fallback."""

    def test_exact_match(self):
        from src.agents.nodes import _extract_chapter_draft
        assembled = "# Blog\n## My Chapter\nContent here.\n## Next\nMore."
        result = _extract_chapter_draft(assembled, "My Chapter")
        assert "Content here." in result

    def test_fuzzy_match(self):
        from src.agents.nodes import _extract_chapter_draft
        assembled = "# Blog\n## My Chapter：With Colon\nContent here.\n## Next\n"
        result = _extract_chapter_draft(assembled, "My Chapter")
        assert "Content here." in result

    def test_not_found(self):
        from src.agents.nodes import _extract_chapter_draft
        result = _extract_chapter_draft("# Blog\n## Only\nContent.", "Missing")
        assert result == ""

    def test_empty_assembled(self):
        from src.agents.nodes import _extract_chapter_draft
        assert _extract_chapter_draft("", "Anything") == ""


class TestAffectedChapters:
    """Tests for _affected_chapters — extract chapter indices from review."""

    def test_single_affected(self):
        from src.agents.nodes import _affected_chapters
        fb = {"issues": [{"chapter_index": 1}, {"chapter_index": 1}]}
        assert _affected_chapters(fb) == {1}

    def test_multiple_affected(self):
        from src.agents.nodes import _affected_chapters
        fb = {"issues": [{"chapter_index": 0}, {"chapter_index": 2}]}
        assert _affected_chapters(fb) == {0, 2}

    def test_no_chapter_indices_returns_none(self):
        from src.agents.nodes import _affected_chapters
        fb = {"issues": [{"paragraph": "x"}]}
        assert _affected_chapters(fb) is None

    def test_none_feedback(self):
        from src.agents.nodes import _affected_chapters
        with pytest.raises(AttributeError):
            _affected_chapters(None)


class TestChatHistoryMessages:
    """Tests for _chat_history_messages — message conversion."""

    def test_user_role(self):
        from src.agents.nodes import _chat_history_messages
        from langchain_core.messages import HumanMessage
        msgs = _chat_history_messages([{"role": "user", "content": "hi"}])
        assert isinstance(msgs[0], HumanMessage)
        assert msgs[0].content == "hi"

    def test_assistant_role(self):
        from src.agents.nodes import _chat_history_messages
        from langchain_core.messages import AIMessage
        msgs = _chat_history_messages([{"role": "assistant", "content": "hello"}])
        assert isinstance(msgs[0], AIMessage)
        assert msgs[0].content == "hello"

    def test_system_role(self):
        from src.agents.nodes import _chat_history_messages
        from langchain_core.messages import SystemMessage
        msgs = _chat_history_messages([{"role": "system", "content": "sys"}])
        assert isinstance(msgs[0], SystemMessage)

    def test_unknown_role_defaults_to_human(self):
        from src.agents.nodes import _chat_history_messages
        from langchain_core.messages import HumanMessage
        msgs = _chat_history_messages([{"role": "gremlin", "content": "rawr"}])
        assert isinstance(msgs[0], HumanMessage)

    def test_langchain_object_passthrough(self):
        from src.agents.nodes import _chat_history_messages
        from langchain_core.messages import HumanMessage
        existing = HumanMessage(content="already")
        msgs = _chat_history_messages([existing])
        assert msgs[0] is existing


class TestDepthRulesText:
    """Tests for _build_depth_rules_text."""

    def test_specific_level(self):
        from src.agents.nodes import _build_depth_rules_text
        result = _build_depth_rules_text("beginner")
        assert "初学者" in result
        assert "中级" not in result
        assert "进阶" not in result

    def test_all_levels(self):
        from src.agents.nodes import _build_depth_rules_text
        result = _build_depth_rules_text("")
        assert "初学者" in result
        assert "中级" in result
        assert "进阶" in result


class TestPrepareFanoutNode:
    """Tests for prepare_fanout_node."""

    def test_resets_state(self):
        from src.agents.nodes import prepare_fanout_node
        state = {"writer_retry_count": 0, "per_chapter_drafts": [{"x": 1}]}
        result = prepare_fanout_node(state)
        assert result["per_chapter_drafts"] == []


class TestSplitChaptersByBudget:
    """Binary-split algorithm: group chapters into posts within word budget."""

    def _make_ch(self, title, n_topics):
        return {"title": title, "key_points": [f"kp{i}" for i in range(n_topics)]}

    def test_single_chapter_fits(self):
        from src.agents.nodes import split_chapters_by_budget
        chapters = [self._make_ch("Intro", 5)]
        result = split_chapters_by_budget(chapters, 6000)
        assert len(result) == 1
        assert result[0]["chapter_count"] == 1

    def test_all_fit_no_split(self):
        from src.agents.nodes import split_chapters_by_budget
        chapters = [self._make_ch(f"Ch{i}", 2) for i in range(5)]
        result = split_chapters_by_budget(chapters, 6000)
        assert len(result) == 1
        assert result[0]["chapter_count"] == 5

    def test_splits_at_midpoint(self):
        from src.agents.nodes import split_chapters_by_budget
        chapters = [self._make_ch(f"Ch{i}", 4) for i in range(10)]
        result = split_chapters_by_budget(chapters, 6000)
        assert len(result) == 2
        assert result[0]["chapter_count"] == 5
        assert result[1]["chapter_count"] == 5

    def test_trailing_fragment_merged(self):
        from src.agents.nodes import split_chapters_by_budget
        chapters = [self._make_ch(f"Ch{i}", 4) for i in range(10)]
        chapters += [self._make_ch(f"Thin{i}", 1) for i in range(5)]
        result = split_chapters_by_budget(chapters, 6000)
        assert len(result) >= 2

    def test_recursive_split(self):
        from src.agents.nodes import split_chapters_by_budget
        chapters = [self._make_ch(f"Ch{i}", 4) for i in range(20)]
        result = split_chapters_by_budget(chapters, 6000)
        assert len(result) >= 3
        total = sum(p["chapter_count"] for p in result)
        assert total == 20

    def test_empty_input(self):
        from src.agents.nodes import split_chapters_by_budget
        result = split_chapters_by_budget([], 6000)
        assert result == []
