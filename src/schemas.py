"""Pydantic data contracts for inter-agent communication."""
import json
from pydantic import BaseModel, Field, field_validator, model_validator, ValidationError
from typing import Optional


# ================================================================
# Shared: level normalization (single source of truth)
# ================================================================

def normalize_level_str(v: str) -> str:
    """Normalize fuzzy level descriptors to beginner/intermediate/advanced."""
    import re
    v_clean = v.strip()
    if v_clean in ("beginner", "intermediate", "advanced"):
        return v_clean
    fuzzy_beginner = {"还行吧", "一般", "会一点", "了解一点", "不太懂", "刚入门",
                      "新手", "小白", "初学", "入门", "rookie", "junior", "一点点"}
    if v_clean in fuzzy_beginner:
        return "beginner"
    senior_kw = ["专家", "精通", "资深", "架构师", "高级", "advanced", "senior"]
    if any(kw in v_clean for kw in senior_kw):
        return "advanced"
    if "多年" in v_clean:
        return "advanced"
    m = re.search(r"(\d+)\s*年", v_clean)
    if m and int(m.group(1)) >= 5:
        return "advanced"
    exp_kw = ["做过", "比较熟", "熟悉", "熟练掌握", "中级", "中等"]
    if any(kw in v_clean for kw in exp_kw):
        return "intermediate"
    if re.search(r"\d+\s*年.*(经验|开发)", v_clean):
        return "intermediate"
    return "beginner"


# ================================================================
# Agent 1: NeedsAlignment → LearnerProfile
# ================================================================

class LearnerProfile(BaseModel):
    """Produced by: NeedsAlignment. Consumed by: KnowledgeTreeBuilder."""
    domain: str = Field(..., min_length=1)
    level: str = Field(...)
    goal: str = Field(..., min_length=1)
    time_constraint: Optional[str] = Field(default=None)
    style: str = Field(default="balanced")

    @field_validator("level")
    @classmethod
    def normalize_level(cls, v: str) -> str:
        return normalize_level_str(v)

    @field_validator("style")
    @classmethod
    def normalize_style(cls, v: str) -> str:
        return v if v in ("practical", "theoretical", "balanced") else "balanced"

    @field_validator("domain")
    @classmethod
    def check_domain(cls, v: str) -> str:
        if len(v.strip()) < 1:
            raise ValueError("domain cannot be empty")
        return v.strip()


# ================================================================
# Agent 2: KnowledgeTreeBuilder → KnowledgeTree
# ================================================================

class KnowledgeTree(BaseModel):
    """Produced by: KnowledgeTreeBuilder. Consumed by: ChapterPlanner.

    Ordered list of learning topics. Grouping into blog posts is handled
    by downstream nodes based on word budget — this node maps the field.
    """
    domain: str = Field(default="")
    topics: list[str] = Field(default_factory=list)

    @field_validator("topics")
    @classmethod
    def check_non_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("Knowledge tree must have at least one topic")
        return v


# ================================================================
# Agent 3: ChapterPlanner → ChapterPlan
# ================================================================

class ChapterPlanItem(BaseModel):
    """A single chapter — ChapterPlanner groups topics, Writer adds narrative."""
    title: str = Field(..., min_length=1)
    key_points: list[str] = Field(default_factory=list)


class ChapterPlan(BaseModel):
    """Produced by: ChapterPlanner. Consumed by: Writer."""
    post_title: str = Field(default="")
    chapters: list[ChapterPlanItem] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_has_chapters(self):
        if not self.chapters:
            raise ValueError("ChapterPlan must have at least one chapter")
        return self


# ================================================================
# Agent 4: Writer → Chapter content (string), no structured schema
# ================================================================

# Writer outputs raw Markdown — no Pydantic model needed.
# The raw output is validated by Reviewer.


# ================================================================
# Agent 5: Reviewer → ReviewResult
# ================================================================

class SplitGroupPlan(BaseModel):
    title: str = Field(default="")
    content_range: str = Field(default="", description="如 从开头到「向量检索实战」结束")
    rationale: str = Field(default="")


class SplitSuggestionDetail(BaseModel):
    split_reason: str = Field(default="")
    groups: list[SplitGroupPlan] = Field(default_factory=list)


class ReviewIssue(BaseModel):
    paragraph: str = Field(default="全文")
    type: str = Field(default="")
    severity: str = Field(default="minor")
    description: str = Field(default="")
    suggestion: str = Field(default="")
    split_suggestion: Optional[SplitSuggestionDetail] = Field(default=None)


class ReviewResult(BaseModel):
    """Produced by: Reviewer. Consumed by: Writer (if reject) or final output."""
    action: str = Field(default="accept")
    word_count: int = Field(default=0)
    overall_assessment: str = Field(default="")
    issues: list[ReviewIssue] = Field(default_factory=list)

    @field_validator("action")
    @classmethod
    def check_action(cls, v: str) -> str:
        if v not in ("accept", "reject"):
            raise ValueError(f"action must be 'accept' or 'reject'")
        return v


# ================================================================
# Tool input validation
# ================================================================

class TavilySearchInput(BaseModel):
    query: str = Field(..., min_length=1, max_length=400)
    max_results: int = Field(default=5, ge=1, le=20)


class FetchPageInput(BaseModel):
    url: str = Field(..., min_length=5)
    max_chars: int = Field(default=8000, ge=1, le=50000)

    @field_validator("url")
    @classmethod
    def check_url(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return v


class VectorQueryInput(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)


# ================================================================
# Validation utility
# ================================================================

def validate_or_raise(model_cls: type[BaseModel], data: dict, source: str = "unknown") -> dict:
    """Validate data against a Pydantic model. Raises ValueError on failure."""
    try:
        return model_cls(**data).model_dump()
    except ValidationError as e:
        raise ValueError(
            f"[{source}] schema validation failed: {e}\n"
            f"Expected fields: {list(model_cls.model_fields.keys())}\n"
            f"Got: {json.dumps(data, ensure_ascii=False, indent=2)[:500]}"
        ) from e
