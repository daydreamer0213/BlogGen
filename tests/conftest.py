"""Test fixtures and shared utilities."""
import os
import pytest
from unittest.mock import MagicMock, patch

# Set env vars BEFORE any src module import (config.py validates at import time)
os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")
os.environ.setdefault("TAVILY_API_KEY", "test-key")


# ================================================================
# Optional dependency checks — skip tests gracefully when deps missing
# ================================================================

def _has(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


HAS_LANGGRAPH = _has("langgraph")
HAS_BS4 = _has("bs4")
HAS_LANGCHAIN_CORE = _has("langchain_core")
HAS_CHROMADB = _has("chromadb")
HAS_SENTENCE_TRANSFORMERS = _has("sentence_transformers")
HAS_JINJA2 = _has("jinja2")
HAS_STREAMLIT = _has("streamlit")

needs_langgraph = pytest.mark.skipif(not HAS_LANGGRAPH, reason="langgraph not installed")
needs_bs4 = pytest.mark.skipif(not HAS_BS4, reason="bs4 not installed")
needs_jinja2 = pytest.mark.skipif(not HAS_JINJA2, reason="jinja2 not installed (required by langchain)")
needs_chromadb = pytest.mark.skipif(not HAS_CHROMADB, reason="chromadb not installed")
needs_langchain = pytest.mark.skipif(not HAS_LANGCHAIN_CORE, reason="langchain not installed")
needs_all_deps = pytest.mark.skipif(
    not (HAS_LANGGRAPH and HAS_BS4 and HAS_JINJA2),
    reason="Core dependencies (langgraph, bs4, jinja2) not installed"
)


@pytest.fixture
def sample_learner_profile():
    return {
        "domain": "AI应用开发",
        "level": "beginner",
        "goal": "通过面试",
        "time_constraint": None,
        "style": "balanced",
    }


@pytest.fixture
def sample_knowledge_tree():
    return {
        "domain": "AI应用开发",
        "topics": [
            "检索增强生成：概念、动机、与LLM微调的对比",
            "文档加载与解析：PDF、Markdown、CSV等格式处理",
            "文本分块策略：固定大小 vs 语义分块、参数调优",
            "向量嵌入：embedding模型选型、维度与性能",
            "向量数据库：Chroma、Milvus、Pinecone对比选型",
            "相似度检索：余弦相似度、ANN近似搜索、Top-K",
            "混合检索：BM25+向量、RRF融合排序",
            "Reranker重排序：cross-encoder提升精度",
            "评估与监控：命中率、MRR、延迟监控",
            "生产部署：性能优化、成本控制、更新策略",
        ],
    }


@pytest.fixture
def sample_review_result_accept():
    return {
        "action": "accept",
        "word_count": 12000,
        "overall_assessment": "内容完整，深度合适",
        "issues": [],
    }


@pytest.fixture
def sample_review_result_reject():
    return {
        "action": "reject",
        "word_count": 6000,
        "overall_assessment": "内容偏长，部分章节深度不足",
        "issues": [
            {
                "paragraph": "第3章",
                "type": "大纲对齐",
                "severity": "minor",
                "description": "缺少核心问题的原理剖析",
                "suggestion": "补充机制解释",
            }
        ],
    }


@pytest.fixture
def sample_chapter_plan():
    return {
        "post_title": "RAG入门",
        "chapters": [
            {"title": "ChatGPT为什么需要开卷考试", "key_points": ["LLM局限", "检索增强思想", "三步流程"]},
        ],
    }
