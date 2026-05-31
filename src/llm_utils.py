"""LLM utilities: DeepSeek client factory, JSON extraction, function-call retry.

Key exports:
  - get_llm()     → ChatOpenAI with DeepSeek API, configurable model + timeout
  - get_fast_llm() → ChatOpenAI with LLM_MODEL_FAST (flash) for simple tasks
  - extract_json() → 4-strategy JSON extraction from LLM output (robust to fences)
  - safe_extract_json() → non-raising version, returns default on failure
  - invoke_with_retry() → exponential backoff LLM call with trace wrapper
"""
import json
import re
import time
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool as langchain_tool
from src.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, LLM_MODEL, LLM_MODEL_FAST


def get_llm(model: str = LLM_MODEL, temperature: float = 0.3, timeout: int | None = None) -> ChatOpenAI:
    """Create a DeepSeek-compatible ChatOpenAI instance.

    Attaches BlogGenCallback for real token counting via LangChain callbacks.
    If timeout is None, uses MAX_LLM_TIMEOUT_SEC from config.
    """
    from src.monitor import get_callback
    from src.config import MAX_LLM_TIMEOUT_SEC
    return ChatOpenAI(
        model=model,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=temperature,
        timeout=timeout if timeout is not None else MAX_LLM_TIMEOUT_SEC,
        max_retries=2,
        callbacks=[get_callback()],
    )


def get_fast_llm(temperature: float = 0.3) -> ChatOpenAI:
    """Get the faster/cheaper DeepSeek model for simple tasks."""
    return get_llm(model=LLM_MODEL_FAST, temperature=temperature)


# ============================================================
# JSON extraction — robust across LLM output formats
# ============================================================

def extract_json(text: str) -> dict:
    """Extract JSON from LLM output with multiple fallback strategies.

    LLMs (especially DeepSeek) sometimes wrap JSON in markdown fences,
    add trailing commas, include explanatory text, or use single quotes.
    This handles all common failure modes.
    """
    if not text or not text.strip():
        raise ValueError("empty response from LLM")

    # Strategy 1: Pure JSON (most common when LLM follows instructions)
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Strategy 2: JSON in markdown code block
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Strategy 3: Find first { ... } block
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        candidate = text[brace_start:brace_end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Strategy 4: Fix common LLM JSON mistakes
    if brace_start != -1 and brace_end > brace_start:
        candidate = text[brace_start:brace_end + 1]
        # Remove trailing commas before ] or }
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        # Fix single-quoted JSON keys: 'key':  →  "key":
        candidate = re.sub(r"'([^'\"{}]+?)'\s*:", r'"\1":', candidate)
        # Fix single-quoted JSON values: : 'value'  →  : "value"
        candidate = re.sub(r":\s*'([^'\"{}]+?)'", r': "\1"', candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Unable to extract JSON from LLM response: {text[:200]}...")


def safe_extract_json(text: str, default: dict | None = None) -> dict:
    """Try extract_json, return default on failure instead of raising."""
    try:
        return extract_json(text)
    except (ValueError, json.JSONDecodeError):
        return default if default is not None else {}


# ============================================================
# Retry wrapper for LLM calls
# ============================================================

def invoke_with_retry(llm: ChatOpenAI, messages: list, max_retries: int = 2, base_delay: float = 2.0) -> AIMessage:
    """Invoke LLM with exponential backoff retry. Always traced via trace_llm_call."""
    from src.monitor import trace_llm_call

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return trace_llm_call(
                lambda msgs: llm.invoke(msgs), messages,
                model=getattr(llm, 'model_name', 'unknown'),
            )
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
    raise last_error


# Tool binding is handled by _get_tool_definitions() in nodes.py
# via _run_with_tools(). All agents use the same tavily_search + query_vector_store pair.
