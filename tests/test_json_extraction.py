"""Test JSON extraction robustness — 4 fallback strategies."""
import pytest
from src.llm_utils import extract_json, safe_extract_json


class TestExtractJson:
    def test_pure_json(self):
        result = extract_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_in_markdown_fence(self):
        text = 'Here is json:\n```json\n{"key": "value"}\n```\nDone.'
        assert extract_json(text) == {"key": "value"}

    def test_json_without_language_specifier(self):
        text = '```\n{"key": "value"}\n```'
        assert extract_json(text) == {"key": "value"}

    def test_json_in_brace_block(self):
        text = 'Some text before. {"domain": "AI", "level": "beginner"} Some after.'
        assert extract_json(text) == {"domain": "AI", "level": "beginner"}

    def test_trailing_comma_fix(self):
        text = '{"name": "test", "items": [1, 2, ],}'
        result = extract_json(text)
        assert result["name"] == "test"
        assert result["items"] == [1, 2]

    def test_single_quotes_fix(self):
        text = "{'key': 'value'}"
        result = extract_json(text)
        assert result == {"key": "value"}

    def test_nested_braces(self):
        text = '{"outer": {"inner": [1, 2]}}'
        assert extract_json(text) == {"outer": {"inner": [1, 2]}}

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="empty"):
            extract_json("")

    def test_unparseable_returns_error(self):
        with pytest.raises(ValueError, match="Unable to extract"):
            extract_json("this is just plain text with no json at all")

    def test_safe_extract_returns_default(self):
        assert safe_extract_json("not json", {"fallback": True}) == {"fallback": True}

    def test_safe_extract_returns_empty_dict(self):
        assert safe_extract_json("not json") == {}

    def test_safe_extract_works_normally(self):
        assert safe_extract_json('{"ok": true}') == {"ok": True}

    def test_deepseek_common_pattern(self):
        """DeepSeek often wraps JSON with explanatory text."""
        text = '好的，我已经了解了你的需求:\n{"domain": "AI", "level": "beginner", "goal": "面试"}'
        assert extract_json(text) == {"domain": "AI", "level": "beginner", "goal": "面试"}

    def test_multiple_markdown_blocks_finds_first_json(self):
        text = '```json\n{"first": true}\n```\n```\nsome code\n```'
        assert extract_json(text) == {"first": True}
