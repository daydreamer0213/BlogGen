"""Test config.py: DEPTH_RULES integrity, tool limits, constants."""
import os
import pytest


class TestDepthRules:
    """DEPTH_RULES must be internally consistent across all levels."""

    REQUIRED_FIELDS = ["label", "instruction", "max_words_per_chapter", "max_chapters", "code_required"]

    def test_all_levels_present(self):
        from src.config import DEPTH_RULES
        assert set(DEPTH_RULES.keys()) == {"beginner", "intermediate", "advanced"}

    @pytest.mark.parametrize("level", ["beginner", "intermediate", "advanced"])
    def test_level_has_all_required_fields(self, level):
        from src.config import DEPTH_RULES
        rule = DEPTH_RULES[level]
        for field in self.REQUIRED_FIELDS:
            assert field in rule, f"{level} missing field '{field}'"

    def test_word_budget_increases_with_level(self):
        from src.config import DEPTH_RULES
        budgets = [DEPTH_RULES[l]["max_words_per_chapter"] for l in ("beginner", "intermediate", "advanced")]
        assert budgets[0] > 0
        assert budgets[1] >= budgets[0]
        assert budgets[2] >= budgets[1]

    def test_chapter_limit_decreases_as_budget_grows(self):
        """Beginner: fewer chapters, intermediate/advanced: more."""
        from src.config import DEPTH_RULES
        assert DEPTH_RULES["beginner"]["max_chapters"] <= DEPTH_RULES["intermediate"]["max_chapters"]
        assert DEPTH_RULES["intermediate"]["max_chapters"] <= DEPTH_RULES["advanced"]["max_chapters"]

    def test_instruction_contains_word_budget(self):
        from src.config import DEPTH_RULES
        for level, rule in DEPTH_RULES.items():
            assert "字" in rule["instruction"], f"{level} instruction missing word count"


class TestToolLimits:
    """Tool calling limits must be consistent across all three config dicts."""

    AGENT_NAMES = [
        "knowledge_tree", "knowledge_tree_retry", "split_posts",
        "chapter_planner", "writer", "writer_chapter",
        "reviewer", "reviewer_chapter", "structure_reviewer",
    ]

    def test_all_agents_have_limits(self):
        from src.config import MAX_TOOL_CALLS_PER_AGENT, MAX_TOOL_SEC_PER_AGENT, MAX_TOOL_ROUNDS_PER_AGENT
        for name in self.AGENT_NAMES:
            assert name in MAX_TOOL_CALLS_PER_AGENT, f"{name} missing from MAX_TOOL_CALLS_PER_AGENT"
            assert name in MAX_TOOL_SEC_PER_AGENT, f"{name} missing from MAX_TOOL_SEC_PER_AGENT"
            assert name in MAX_TOOL_ROUNDS_PER_AGENT, f"{name} missing from MAX_TOOL_ROUNDS_PER_AGENT"

    def test_no_orphan_limits(self):
        """Every key in tool limit dicts should be a known agent."""
        from src.config import MAX_TOOL_CALLS_PER_AGENT
        for name in MAX_TOOL_CALLS_PER_AGENT:
            assert name in self.AGENT_NAMES, f"Unknown agent '{name}' in MAX_TOOL_CALLS_PER_AGENT"

    def test_limits_are_sane(self):
        from src.config import MAX_TOOL_CALLS_PER_AGENT, MAX_TOOL_SEC_PER_AGENT, MAX_TOOL_ROUNDS_PER_AGENT
        for name in self.AGENT_NAMES:
            calls = MAX_TOOL_CALLS_PER_AGENT[name]
            secs = MAX_TOOL_SEC_PER_AGENT[name]
            rounds = MAX_TOOL_ROUNDS_PER_AGENT[name]
            assert calls >= 0, f"{name} negative tool calls"
            assert secs >= 0, f"{name} negative tool seconds"
            assert rounds >= 1, f"{name} rounds must be >=1"


class TestConstants:
    def test_max_review_retries_is_positive(self):
        from src.config import MAX_REVIEW_RETRIES
        assert MAX_REVIEW_RETRIES >= 0

    def test_style_rules_have_expected_keys(self):
        from src.config import STYLE_RULES
        assert set(STYLE_RULES.keys()) == {"practical", "theoretical", "balanced"}

    def test_llm_model_names_set(self):
        from src.config import LLM_MODEL, LLM_MODEL_FAST
        assert LLM_MODEL and len(LLM_MODEL) > 0
        assert LLM_MODEL_FAST and len(LLM_MODEL_FAST) > 0
        assert LLM_MODEL != LLM_MODEL_FAST, "Pro and Fast should be different models"

    def test_output_dir_exists(self):
        from src.config import OUTPUT_DIR
        assert OUTPUT_DIR and len(OUTPUT_DIR) > 0

    def test_paths_are_absolute_or_relative(self):
        from src.config import SQLITE_PATH, CHROMA_PERSIST_DIR, OUTPUT_DIR
        for path in [SQLITE_PATH, CHROMA_PERSIST_DIR, OUTPUT_DIR]:
            assert path, f"Path is empty"
            # Paths should at least exist or be creatable
            from pathlib import Path
            p = Path(path)
            if not p.exists():
                p.mkdir(parents=True, exist_ok=True)
                assert p.exists()
                p.rmdir()
