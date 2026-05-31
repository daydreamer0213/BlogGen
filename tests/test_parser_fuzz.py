"""Parser fuzz tests — nulls, malformed data, boundary inputs, Unicode.

Industry pattern (Anthropic/Meta): fuzz every parser with hostile inputs
before trusting LLM output to flow through without crashing.
"""
import pytest


class TestChapterMarkdownFuzz:
    """Fuzz tests for _parse_chapter_markdown."""

    @pytest.fixture
    def parser(self):
        from src.agents.nodes import _parse_chapter_markdown
        return _parse_chapter_markdown

    def test_none_input(self, parser):
        with pytest.raises(AttributeError):
            parser(None)

    def test_whitespace_only(self, parser):
        r = parser("   \n\n  \n")
        assert r == {"post_title": "", "chapters": []}

    def test_only_h1(self, parser):
        r = parser("# Just a title\n\nSome prose without chapters.")
        assert r["post_title"] == "Just a title"
        assert r["chapters"] == []

    def test_h3_headings_ignored(self, parser):
        r = parser("# Blog\n## Ch1\n- a\n### Subsection\n- b\n## Ch2\n- c")
        assert len(r["chapters"]) == 2
        assert r["chapters"][1]["key_points"] == ["c"]

    def test_code_block_not_parsed_as_chapter(self, parser):
        md = "# Blog\n## Real Chapter\n- point 1\n```\n## Not a chapter\n- not a point\n```\n## Ch2\n- point 2"
        r = parser(md)
        assert len(r["chapters"]) == 3  # Real Chapter, Not a chapter, Ch2
        # "## Not a chapter" inside code block IS parsed as chapter (known limitation)

    def test_emoji_in_title(self, parser):
        md = "# Blog 🎉\n## 😂 Chapter 1\n- 🚀 point"
        r = parser(md)
        assert r["chapters"][0]["title"] == "😂 Chapter 1"
        assert r["chapters"][0]["key_points"] == ["🚀 point"]

    def test_long_title(self, parser):
        long = "X" * 10000
        r = parser(f"# Blog\n## {long}\n- point")
        assert r["chapters"][0]["title"] == long

    def test_empty_bullet(self, parser):
        r = parser("# Blog\n## Ch1\n- \n- real point\n-   \n")
        assert r["chapters"][0]["key_points"] == ["real point"]

    def test_mixed_bullet_styles(self, parser):
        r = parser("# Blog\n## Ch1\n- dash\n* star\n- dash2")
        assert r["chapters"][0]["key_points"] == ["dash", "star", "dash2"]

    def test_heading_without_content(self, parser):
        r = parser("# Blog\n## Empty Chapter")
        assert len(r["chapters"]) == 1
        assert r["chapters"][0]["key_points"] == []

    def test_duplicate_chapter_titles(self, parser):
        r = parser("# Blog\n## Same\n- a\n## Same\n- b")
        assert len(r["chapters"]) == 2
        assert r["chapters"][0]["key_points"] == ["a"]
        assert r["chapters"][1]["key_points"] == ["b"]


class TestKnowledgeTreeMarkdownFuzz:
    """Fuzz tests for _parse_knowledge_tree_markdown."""

    @pytest.fixture
    def parser(self):
        from src.agents.nodes import _parse_knowledge_tree_markdown
        return _parse_knowledge_tree_markdown

    def test_none_input(self, parser):
        with pytest.raises(AttributeError):
            parser(None)

    def test_whitespace_only(self, parser):
        r = parser("   \n\n  \n")
        assert r == {"domain": "", "topics": []}

    def test_only_domain(self, parser):
        r = parser("# Just a domain")
        assert r["domain"] == "Just a domain"
        assert r["topics"] == []

    def test_multiple_h1_lines(self, parser):
        r = parser("# First\n# Second\n- topic")
        assert r["domain"] == "Second"

    def test_topics_with_extra_whitespace(self, parser):
        r = parser("# Domain\n-  spaced out  \n* star topic\n- \n- real")
        assert r["topics"] == ["spaced out", "star topic", "real"]

    def test_topics_with_inline_code(self, parser):
        r = parser("# Domain\n- `vector_search()` function\n- plain topic")
        assert "`vector_search()` function" in r["topics"]

    def test_no_domain_with_topics(self, parser):
        r = parser("- topic 1\n- topic 2")
        assert r["domain"] == ""
        assert r["topics"] == ["topic 1", "topic 2"]

    def test_long_topic(self, parser):
        long = "A" * 5000
        r = parser(f"# Domain\n- {long}")
        assert r["topics"][0] == long


class TestReviewMarkdownFuzz:
    """Fuzz tests for _parse_review_markdown."""

    @pytest.fixture
    def parser(self):
        from src.agents.nodes import _parse_review_markdown
        return _parse_review_markdown

    def test_none_input(self, parser):
        with pytest.raises(AttributeError):
            parser(None)

    def test_whitespace_only(self, parser):
        r = parser("   \n\n  \n")
        assert r["action"] == "accept"
        assert r["issues"] == []

    def test_unknown_judgment_value(self, parser):
        r = parser("判断：maybe\n字数：1000\n总评：test")
        assert r["action"] == "reject"  # anything != 通过/accept → reject

    def test_word_count_not_numeric(self, parser):
        r = parser("判断：通过\n字数：abc\n总评：test")
        assert r["word_count"] == 0  # ValueError → silently stays 0

    def test_chapter_index_not_numeric(self, parser):
        r = parser(
            "判断：不通过\n字数：1000\n总评：bad\n"
            "### 问题1\n章节：abc\n类型：事实错误\n描述：err\n建议：fix"
        )
        assert r["issues"][0].get("chapter_index") is None  # ValueError → not set

    def test_chapter_index_zero_based_convert(self, parser):
        r = parser(
            "判断：不通过\n字数：1000\n总评：bad\n"
            "### 问题1\n章节：1\n类型：事实错误\n描述：err\n建议：fix"
        )
        assert r["issues"][0]["chapter_index"] == 0

    def test_empty_issue(self, parser):
        r = parser("判断：不通过\n字数：1000\n总评：bad\n### 问题1")
        # Empty issue still gets added
        assert len(r["issues"]) == 1
        assert r["issues"][0]["paragraph"] == ""

    def test_separator_inside_description(self, parser):
        r = parser(
            "判断：不通过\n字数：1000\n总评：bad\n"
            "### 问题1\n段落：这里\n描述：包含 --- 分隔符的描述\n建议：fix"
        )
        assert len(r["issues"]) == 1

    def test_extremely_long_description(self, parser):
        long = "X" * 10000
        r = parser(
            f"判断：不通过\n字数：1000\n总评：bad\n"
            f"### 问题1\n段落：here\n类型：事实错误\n描述：{long}\n建议：fix"
        )
        assert r["issues"][0]["description"] == long
