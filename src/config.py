import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")


def _require(key: str) -> str:
    val = os.getenv(key, "")
    if not val:
        raise ValueError(f"{key} is not set in .env file")
    if val.startswith("sk-placeholder") or val.startswith("your-"):
        raise ValueError(f"{key} appears to be a placeholder value, please set a real key")
    return val


# LLM
DEEPSEEK_API_KEY = _require("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
LLM_MODEL = "deepseek-v4-pro"
LLM_MODEL_FAST = "deepseek-v4-flash"

# Feature flags
ENABLE_NETWORK_SEARCH = os.getenv("ENABLE_NETWORK_SEARCH", "false").lower() in ("1", "true", "yes")

# Tavily — only required when network search is enabled
TAVILY_API_KEY = _require("TAVILY_API_KEY") if ENABLE_NETWORK_SEARCH else ""

# BGE
BGE_EMBEDDING_MODEL = os.getenv("BGE_EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
BGE_RERANKER_MODEL = os.getenv("BGE_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

# Chroma
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(Path(__file__).parent.parent / "data" / "chroma"))

# SQLite
SQLITE_PATH = os.getenv("SQLITE_PATH", str(Path(__file__).parent.parent / "data" / "checkpoints.db"))

# Output
OUTPUT_DIR = os.getenv("OUTPUT_DIR", str(Path(__file__).parent.parent / "outputs"))

# Constraints
MAX_REVIEW_RETRIES = 2       # Max times Writer can be called back for fixes
RECENCY_CUTOFF_YEARS = 1

# Tool calling limits (per agent) — conservative, let pipeline complete first
MAX_LLM_TIMEOUT_SEC = 120    # Single LLM HTTP request timeout
MAX_TOOL_CALLS_PER_AGENT = { # Max total tool calls per _run_with_tools
    "knowledge_tree": 1,
    "knowledge_tree_retry": 0,
    "split_posts": 0,            # Split planning: no tools needed
    "chapter_planner": 1,
    "writer": 1,
    "writer_chapter": 1,
    "reviewer": 1,
    "reviewer_chapter": 0,       # Per-chapter: no tools, code checks coverage
    "structure_reviewer": 0,     # Structure check: no tools
}
MAX_TOOL_SEC_PER_AGENT = {   # Cumulative timeout
    "knowledge_tree": 180,
    "split_posts": 60,           # Quick split plan, Flash
    "chapter_planner": 120,
    "writer": 300,
    "writer_chapter": 300,
    "reviewer": 120,
    "reviewer_chapter": 90,
    "structure_reviewer": 90,
}
MAX_TOOL_ROUNDS_PER_AGENT = {  # Max LLM invocations per _run_with_tools
    "knowledge_tree": 2,
    "split_posts": 1,            # Single LLM call to plan splits
    "chapter_planner": 2,
    "writer": 2,
    "writer_chapter": 1,
    "reviewer": 1,
    "reviewer_chapter": 1,      # Single pass, checklist-based
    "structure_reviewer": 1,
}

# Depth rules — maps learner level to instruction modifiers
DEPTH_RULES = {
    "beginner": {
        "label": "初学者",
        "instruction": (
            "目标读者是初学者。每章控制在 1500-1800 字，整篇 5000-7000 字。"
            "每章深入覆盖 2-3 个最重要知识点，不要试图覆盖所有。"
            "每个概念都要有生活化类比，避免数学公式，"
            "注重'是什么'和'怎么用'。每章至少一个简短伪代码示例（≤30行）+ 运行结果。"
        ),
        "max_words_per_chapter": 1800,
        "max_chapters": 6,
        "code_required": True,
    },
    "intermediate": {
        "label": "中级",
        "instruction": (
            "目标读者有中级基础。每章控制在 1500-2000 字，整篇 6000-8000 字。"
            "每章深入覆盖 2-4 个知识点，聚焦深度而非广度。"
            "适当引入公式推导和方案对比，"
            "拓展题为开放式设计题，鼓励读者自己探索。"
        ),
        "max_words_per_chapter": 2000,
        "max_chapters": 8,
        "code_required": True,
    },
    "advanced": {
        "label": "进阶",
        "instruction": (
            "目标读者是进阶开发者。每章控制在 1500-1800 字，整篇 6000-8000 字。"
            "可以深入论文精读、源码分析、"
            "性能benchmark对比。拓展题为研究性问题。"
        ),
        "max_words_per_chapter": 2000,
        "max_chapters": 10,
        "code_required": False,
    },
}

# Blog style preferences
STYLE_RULES = {
    "practical": "偏实战，重代码示例和可操作的实践指南",
    "theoretical": "偏理论深挖，重原理推导和学术背景",
    "balanced": "实战与理论平衡，既有原理也有代码",
}
