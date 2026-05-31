"""Test tool input validation."""
import pytest
from src.schemas import (
    TavilySearchInput, FetchPageInput, VectorQueryInput,
    validate_or_raise,
)


class TestTavilySearchInput:
    def test_valid_input(self):
        v = validate_or_raise(TavilySearchInput, {"query": "RAG tutorial", "max_results": 5}, "test")
        assert v["query"] == "RAG tutorial"
        assert v["max_results"] == 5

    def test_empty_query_raises(self):
        with pytest.raises(ValueError):
            TavilySearchInput(query="", max_results=5)

    def test_max_results_clamped(self):
        v = TavilySearchInput(query="test", max_results=20)
        assert v.max_results == 20  # within 1-20 range


class TestFetchPageInput:
    def test_valid_url(self):
        v = FetchPageInput(url="https://example.com")
        assert v.url == "https://example.com"
        assert v.max_chars == 8000

    def test_no_protocol_raises(self):
        with pytest.raises(ValueError):
            FetchPageInput(url="example.com")

    def test_max_chars_default(self):
        v = FetchPageInput(url="https://example.com")
        assert hasattr(v, "max_chars")


class TestVectorQueryInput:
    def test_valid_input(self):
        v = VectorQueryInput(query="RAG", top_k=5)
        assert v.query == "RAG"
        assert v.top_k == 5

    def test_empty_query_raises(self):
        with pytest.raises(ValueError):
            VectorQueryInput(query="", top_k=5)
