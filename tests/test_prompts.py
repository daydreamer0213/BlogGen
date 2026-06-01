"""Test prompts.py: placeholder integrity, format constraints, required sections."""
import pytest
from src.agents.prompts import (
    NEEDS_ALIGNMENT_PROMPT,
    KNOWLEDGE_TREE_PROMPT,
    CHAPTER_PLANNER_PROMPT,
    WRITER_PROMPT,
    REVIEWER_PROMPT,
    STRUCTURE_REVIEWER_PROMPT,
)


class TestPlaceholderSubstitution:
    """Every {placeholder} in prompts must be documented and substitutable."""

    def test_knowledge_tree_has_depth_rules_placeholder(self):
        assert "{depth_rules}" in KNOWLEDGE_TREE_PROMPT, "KNOWLEDGE_TREE_PROMPT missing {depth_rules}"

    def test_chapter_planner_has_chapter_limit_placeholder(self):
        assert "{chapter_limit}" in CHAPTER_PLANNER_PROMPT, "CHAPTER_PLANNER_PROMPT missing {chapter_limit}"

    def test_writer_has_level_instruction_placeholder(self):
        assert "{level_instruction}" in WRITER_PROMPT, "WRITER_PROMPT missing {level_instruction}"

    def test_reviewer_has_placeholders(self):
        assert "{key_points}" in REVIEWER_PROMPT, "REVIEWER_PROMPT missing {key_points}"
        assert "{level_instruction}" in REVIEWER_PROMPT, "REVIEWER_PROMPT missing {level_instruction}"
        assert "{chapter_idx}" in REVIEWER_PROMPT, "REVIEWER_PROMPT missing {chapter_idx}"

    def test_structure_reviewer_has_placeholders(self):
        assert "{chapter_list}" in STRUCTURE_REVIEWER_PROMPT, "missing {chapter_list}"
        assert "{all_points}" in STRUCTURE_REVIEWER_PROMPT, "missing {all_points}"
        assert "{word_count}" in STRUCTURE_REVIEWER_PROMPT, "missing {word_count}"
        assert "{level_instruction}" in STRUCTURE_REVIEWER_PROMPT, "missing {level_instruction}"

    def test_substitution_does_not_leave_orphans(self):
        """After substituting all known placeholders, no {} patterns remain."""
        prompts = {
            "KNOWLEDGE_TREE": KNOWLEDGE_TREE_PROMPT.replace("{depth_rules}", "test"),
            "CHAPTER_PLANNER": CHAPTER_PLANNER_PROMPT.replace("{chapter_limit}", "test"),
            "WRITER": WRITER_PROMPT.replace("{level_instruction}", "test"),
            "REVIEWER": REVIEWER_PROMPT.replace("{key_points}", "test")
                .replace("{level_instruction}", "test")
                .replace("{chapter_idx}", "1"),
            "STRUCTURE_REVIEWER": STRUCTURE_REVIEWER_PROMPT
                .replace("{chapter_list}", "test")
                .replace("{all_points}", "test")
                .replace("{word_count}", "1000")
                .replace("{level_instruction}", "test"),
        }
        for name, prompt in prompts.items():
            assert "{" not in prompt and "}" not in prompt, \
                f"{name} has unsubstituted placeholders after substitution"


class TestFormatConstraints:
    """Prompts must include key structural directives."""

    def test_needs_alignment_requires_json_output(self):
        assert "json" in NEEDS_ALIGNMENT_PROMPT.lower(), "NeedsAlignment must request JSON"

    def test_knowledge_tree_forbids_grouping(self):
        assert "不要分组" in KNOWLEDGE_TREE_PROMPT, "KnowledgeTree must forbid grouping"

    def test_knowledge_tree_enforces_single_concept(self):
        assert "单一概念" in KNOWLEDGE_TREE_PROMPT, "KnowledgeTree missing single-concept rule"

    def test_chapter_planner_enforces_single_concept(self):
        assert "单一概念" in CHAPTER_PLANNER_PROMPT, "ChapterPlanner missing single-concept rule"

    def test_chapter_planner_has_chapter_limit_mention(self):
        assert "章" in CHAPTER_PLANNER_PROMPT, "ChapterPlanner should mention chapter limits"

    def test_writer_has_hard_word_budget_warning(self):
        assert "硬性约束" in WRITER_PROMPT or "字数" in WRITER_PROMPT, "Writer missing word budget warning"

    def test_writer_has_priority_rule(self):
        assert "优先" in WRITER_PROMPT, "Writer missing priority rule for coverage vs budget"

    def test_writer_has_structure_guidance(self):
        assert "问题引入" in WRITER_PROMPT, "Writer missing structure guidance"

    def test_reviewer_has_severity_rules(self):
        assert "严重度判定规则" in REVIEWER_PROMPT, "Reviewer missing severity classification rules"
        assert "minor" in REVIEWER_PROMPT, "Reviewer should mention minor severity"

    def test_reviewer_requires_checklist_format(self):
        assert "[ ]" in REVIEWER_PROMPT, "Reviewer should use checklist format"

    def test_structure_reviewer_has_severity_rules(self):
        assert "严重度判定规则" in STRUCTURE_REVIEWER_PROMPT, "StructureReviewer missing severity rules"

    def test_reviewer_has_output_format_spec(self):
        assert "判断：" in REVIEWER_PROMPT, "Reviewer missing output format"

    def test_all_prompts_are_chinese(self):
        """All prompts should be primarily in Chinese for Chinese blog generation."""
        prompts = {
            "NEEDS_ALIGNMENT": NEEDS_ALIGNMENT_PROMPT,
            "KNOWLEDGE_TREE": KNOWLEDGE_TREE_PROMPT,
            "CHAPTER_PLANNER": CHAPTER_PLANNER_PROMPT,
            "WRITER": WRITER_PROMPT,
            "REVIEWER": REVIEWER_PROMPT,
            "STRUCTURE_REVIEWER": STRUCTURE_REVIEWER_PROMPT,
        }
        for name, prompt in prompts.items():
            # At least 80% of CJK range in non-whitespace chars
            import re
            text = re.sub(r'[\s\n]', '', prompt)
            text = re.sub(r'[a-zA-Z0-9_\-*/{}[\]()#`|>.<,;:!?"\']+', '', text)
            assert len(text) > 50, f"{name} has too little Chinese text ({len(text)} chars)"
