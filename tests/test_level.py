"""Test level normalization logic."""
import pytest
from src.agents.nodes import _normalize_level


class TestLevelNormalization:

    @pytest.mark.parametrize("raw,expected", [
        # Standard values pass through
        ("beginner", "beginner"),
        ("intermediate", "intermediate"),
        ("advanced", "advanced"),
        # Fuzzy → beginner
        ("还行吧", "beginner"),
        ("一般", "beginner"),
        ("会一点", "beginner"),
        ("不太懂", "beginner"),
        ("刚入门", "beginner"),
        ("新手", "beginner"),
        ("小白", "beginner"),
        ("初学者", "beginner"),
        ("入门", "beginner"),
        ("rookie", "beginner"),
        ("junior", "beginner"),
        # Experience regex → intermediate (<5 years)
        ("有3年Python经验", "intermediate"),
        # 5+ years → advanced
        ("5年开发经验", "advanced"),
        ("做过几个项目", "intermediate"),
        ("比较熟悉RAG", "intermediate"),
        ("熟练掌握", "intermediate"),
        ("中级水平", "intermediate"),
        ("中等", "intermediate"),
        # Senior keywords → advanced
        ("精通", "advanced"),
        ("资深工程师", "advanced"),
        ("架构师", "advanced"),
        ("高级开发", "advanced"),
        ("advanced", "advanced"),
        ("senior", "advanced"),
        ("专家", "advanced"),
        # N-year pattern → advanced
        ("10年经验", "advanced"),
        ("多年经验", "advanced"),
        # Unknown → default beginner
        ("random text", "beginner"),
        ("", "beginner"),
        ("懂一点但不多", "beginner"),
    ])
    def test_normalize(self, raw, expected):
        assert _normalize_level(raw) == expected

    def test_whitespace_handling(self):
        assert _normalize_level("  中级水平  ") == "intermediate"
