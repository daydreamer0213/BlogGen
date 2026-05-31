"""Test Pydantic schema validation for all data contracts."""
import pytest
from src.schemas import (
    LearnerProfile, KnowledgeTree,
    ChapterPlanItem, ChapterPlan, ReviewIssue, ReviewResult,
    SplitGroupPlan, SplitSuggestionDetail,
    validate_or_raise,
)


class TestLearnerProfile:
    def test_valid_profile(self):
        p = LearnerProfile(domain="AI", level="beginner", goal="面试")
        d = p.model_dump()
        assert d["style"] == "balanced"
        assert d["time_constraint"] is None

    def test_empty_domain_raises(self):
        with pytest.raises(ValueError):
            LearnerProfile(domain="", level="beginner", goal="test")

    def test_fuzzy_level_normalized(self):
        p = LearnerProfile(domain="AI", level="还行吧", goal="面试")
        assert p.level == "beginner"

    def test_experience_level_normalized(self):
        p = LearnerProfile(domain="AI", level="有5年经验", goal="面试")
        assert p.level == "advanced"  # 5+ 年 → advanced

    def test_invalid_style_defaults(self):
        p = LearnerProfile(domain="AI", level="beginner", goal="test", style="unknown")
        assert p.style == "balanced"


class TestKnowledgeTree:
    def test_valid_topic_list(self, sample_knowledge_tree):
        tree = KnowledgeTree(**sample_knowledge_tree)
        d = tree.model_dump()
        assert len(d["topics"]) == 10
        assert d["topics"][0] == "检索增强生成：概念、动机、与LLM微调的对比"

    def test_empty_topics_raises(self):
        with pytest.raises(ValueError):
            KnowledgeTree(domain="test", topics=[])


class TestChapterPlan:
    def test_valid_plan(self, sample_chapter_plan):
        plan = ChapterPlan(**sample_chapter_plan)
        assert plan.post_title == "RAG入门"
        assert len(plan.chapters) == 1

    def test_chapter_item(self):
        ch = ChapterPlanItem(title="Test", key_points=["a", "b"])
        assert ch.title == "Test"
        assert ch.key_points == ["a", "b"]

    def test_empty_title_raises(self):
        with pytest.raises(ValueError):
            ChapterPlanItem(title="", key_points=["a"])

    def test_empty_chapters_raises(self):
        with pytest.raises(ValueError, match="must have at least one chapter"):
            ChapterPlan(post_title="test", chapters=[])

    def test_empty_plan_dict_raises(self):
        with pytest.raises(ValueError):
            ChapterPlan(**{})


class TestReviewResult:
    def test_accept(self, sample_review_result_accept):
        r = ReviewResult(**sample_review_result_accept)
        assert r.action == "accept"

    def test_reject(self, sample_review_result_reject):
        r = ReviewResult(**sample_review_result_reject)
        assert r.action == "reject"
        assert len(r.issues) == 1
        assert r.issues[0].type == "大纲对齐"

    def test_split_group_content_range(self):
        sg = SplitGroupPlan(
            title="第一篇",
            content_range="从开头到检索实战",
            rationale="pipeline",
        )
        d = sg.model_dump()
        assert "content_range" in d
        assert "chapters" not in d  # old field removed

    def test_invalid_action_raises(self):
        with pytest.raises(ValueError):
            ReviewResult(action="maybe", word_count=100, overall_assessment="", issues=[])


class TestValidateOrRaise:
    def test_valid_data(self):
        result = validate_or_raise(LearnerProfile, {"domain": "AI", "level": "beginner", "goal": "test"}, "test")
        assert result["style"] == "balanced"

    def test_invalid_data_raises(self):
        with pytest.raises(ValueError, match="schema validation"):
            validate_or_raise(LearnerProfile, {"domain": "", "level": "beginner", "goal": "test"}, "test")


class TestFetchPageInput:
    def test_max_chars_default(self):
        from src.schemas import FetchPageInput
        v = FetchPageInput(url="https://example.com")
        assert v.max_chars == 8000

    def test_max_chars_custom(self):
        from src.schemas import FetchPageInput
        v = FetchPageInput(url="https://example.com", max_chars=500)
        assert v.max_chars == 500

    def test_max_chars_validated(self):
        from src.schemas import FetchPageInput
        v = validate_or_raise(FetchPageInput, {"url": "https://x.com", "max_chars": 1000}, "test")
        assert v["max_chars"] == 1000


class TestNormalizeLevelStr:
    def test_shared_function(self):
        from src.schemas import normalize_level_str
        assert normalize_level_str("新手") == "beginner"
        assert normalize_level_str("beginner") == "beginner"
        assert normalize_level_str("精通") == "advanced"
        assert normalize_level_str("熟悉") == "intermediate"

    def test_none_input(self):
        from src.schemas import normalize_level_str
        with pytest.raises(AttributeError):
            normalize_level_str(None)

    def test_four_years_is_intermediate(self):
        from src.schemas import normalize_level_str
        assert normalize_level_str("4年经验") == "intermediate"

    def test_boundary_five_years_is_advanced(self):
        from src.schemas import normalize_level_str
        assert normalize_level_str("5年经验") == "advanced"


class TestSchemaBoundaries:
    """Boundary value analysis for all Pydantic schemas."""

    def test_tavily_query_max_length(self):
        from src.schemas import TavilySearchInput
        v = TavilySearchInput(query="a" * 400)
        assert len(v.query) == 400

    def test_tavily_query_over_max(self):
        from src.schemas import TavilySearchInput
        with pytest.raises(ValueError):
            TavilySearchInput(query="a" * 401, max_results=10)

    def test_vector_top_k_min(self):
        from src.schemas import VectorQueryInput
        v = VectorQueryInput(query="test", top_k=1)
        assert v.top_k == 1

    def test_vector_top_k_max(self):
        from src.schemas import VectorQueryInput
        v = VectorQueryInput(query="test", top_k=20)
        assert v.top_k == 20

    def test_vector_top_k_zero_raises(self):
        from src.schemas import VectorQueryInput
        with pytest.raises(ValueError):
            VectorQueryInput(query="test", top_k=0)

    def test_vector_top_k_over_max(self):
        from src.schemas import VectorQueryInput
        with pytest.raises(ValueError):
            VectorQueryInput(query="test", top_k=21)

    def test_fetch_max_chars_min(self):
        from src.schemas import FetchPageInput
        v = FetchPageInput(url="https://example.com", max_chars=1)
        assert v.max_chars == 1

    def test_fetch_max_chars_max(self):
        from src.schemas import FetchPageInput
        v = FetchPageInput(url="https://example.com", max_chars=50000)
        assert v.max_chars == 50000

    def test_fetch_max_chars_zero_raises(self):
        from src.schemas import FetchPageInput
        with pytest.raises(ValueError):
            FetchPageInput(url="https://example.com", max_chars=0)

    def test_fetch_max_chars_over_max(self):
        from src.schemas import FetchPageInput
        with pytest.raises(ValueError):
            FetchPageInput(url="https://example.com", max_chars=50001)

    def test_chapter_item_empty_key_points_ok(self):
        from src.schemas import ChapterPlanItem
        v = ChapterPlanItem(title="Test", key_points=[])
        assert v.key_points == []

    def test_learner_profile_whitespace_only_domain(self):
        from src.schemas import LearnerProfile
        v = LearnerProfile(domain="  AI  ", level="beginner", goal="test")
        assert v.domain == "AI"  # stripped

    def test_learner_profile_valid_styles(self):
        from src.schemas import LearnerProfile
        for style in ("practical", "theoretical", "balanced"):
            v = LearnerProfile(domain="AI", level="beginner", goal="test", style=style)
            assert v.style == style

    def test_review_result_empty_issues_ok(self):
        from src.schemas import ReviewResult
        v = ReviewResult(action="accept", word_count=1000, overall_assessment="ok", issues=[])
        assert v.issues == []

    def test_normalize_level_str_underscore_handling(self):
        from src.schemas import normalize_level_str
        # empty/whitespace → beginner
        assert normalize_level_str("") == "beginner"
        assert normalize_level_str("   ") == "beginner"

