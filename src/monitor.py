"""Agent monitoring: per-node token tracking + timing.

Uses LangChain BaseCallbackHandler for:
  - Real token counts (on_llm_end)
  - Per-node timing (on_chain_start / on_chain_end)
Writes structured JSONL logs to data/logs.jsonl — one entry per node execution.
"""
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from langchain_core.callbacks import BaseCallbackHandler

LOG_FILE = Path(__file__).parent.parent / "data" / "logs.jsonl"
logger = logging.getLogger("BlogGen")

# Rolling buffer for the sidebar
_session_logs: list[dict] = []
_token_total: int = 0
_current_agent: str = ""


def reset_session():
    global _session_logs, _token_total, _current_agent
    _session_logs = []
    _token_total = 0
    _current_agent = ""


def get_session_summary() -> dict:
    return {
        "agent_count": len(_session_logs),
        "total_tokens": _token_total,
        "current_agent": _current_agent,
        "recent": _session_logs[-5:] if _session_logs else [],
    }


def _node_to_agent(node_name: str) -> str:
    """Map LangGraph node names to readable agent names for logging."""
    _MAP = {
        "needs_alignment": "NeedsAlignment",
        "knowledge_tree": "KnowledgeTree",
        "chapter_planner": "ChapterPlanner",
        "writer_batch": "Writer",
        "write_chapter": "Writer",
        "assembler": "Assembler",
        "tier1_check": "Tier1Check",
        "review_batch": "Reviewer",
        "review_chapter": "Reviewer",
        "assemble_reviews": "Reviewer",
        "next_post": "FlowControl",
        "split_posts": "ChapterPlanner",
    }
    return _MAP.get(node_name, node_name)


# ================================================================
# LangChain Callback: per-node timing + per-LLM token tracking
# ================================================================

class BlogGenMonitorCallback(BaseCallbackHandler):
    """Per-node monitoring callback attached to graph.invoke() config.

    Uses on_chain_start/on_chain_end for per-node timing.
    Uses on_llm_start/on_llm_end for per-call token tracking.
    Writes a log entry for every node that executes.
    """

    def __init__(self):
        self._node_starts: dict[str, float] = {}
        self._node_names: dict[str, str] = {}
        self._current_llm_calls: list[dict] = []
        self._current_tool_calls: list[dict] = []
        self._node_entries: list[dict] = []
        self._top_run_id: str | None = None

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, **kwargs):
        name = serialized.get("name", "unknown") if serialized else "unknown"
        # First call is the graph itself
        if self._top_run_id is None:
            self._top_run_id = run_id
            return
        # Only track direct children of the graph
        if parent_run_id is None or parent_run_id != self._top_run_id:
            return
        # Skip obviously non-node chains
        if name in ("RunnableSequence", "RunnableLambda", "RunnableCallable", "<lambda>"):
            # But DO track if it has a meaningful name from the graph
            pass
        self._node_starts[run_id] = time.time()
        self._node_names[run_id] = name
        self._current_llm_calls = []
        self._current_tool_calls = []

    def on_chain_end(self, outputs, *, run_id, parent_run_id=None, **kwargs):
        name = self._node_names.pop(run_id, None)
        t_start = self._node_starts.pop(run_id, None)
        if name is None or t_start is None:
            return
        latency = (time.time() - t_start) * 1000
        _write_log({
            "timestamp": datetime.now().isoformat(),
            "agent": _node_to_agent(name),
            "llm_calls": list(self._current_llm_calls),
            "tool_calls": list(self._current_tool_calls),
            "total_latency_ms": round(latency, 1),
            "error": None,
        })

    def on_chain_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        name = self._node_names.pop(run_id, None)
        t_start = self._node_starts.pop(run_id, None)
        if name is None or t_start is None:
            return
        latency = (time.time() - t_start) * 1000
        _write_log({
            "timestamp": datetime.now().isoformat(),
            "agent": _node_to_agent(name),
            "llm_calls": list(self._current_llm_calls),
            "tool_calls": list(self._current_tool_calls),
            "total_latency_ms": round(latency, 1),
            "error": str(error)[:300],
        })

    # ---- LLM token tracking ----

    def on_llm_end(self, response, **kwargs):
        latency = (time.time() - getattr(self, '_llm_t_start', time.time())) * 1000
        prompt_tokens = 0
        completion_tokens = 0
        model = "unknown"

        try:
            if hasattr(response, "llm_output") and response.llm_output:
                usage = response.llm_output.get("token_usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
            if not prompt_tokens and hasattr(response, "generations"):
                gen = response.generations[0][0]
                if hasattr(gen, "generation_info") and gen.generation_info:
                    usage = gen.generation_info.get("usage_metadata", {})
                    prompt_tokens = usage.get("input_tokens", 0)
                    completion_tokens = usage.get("output_tokens", 0)
            if hasattr(response, "generations") and response.generations:
                msg = response.generations[0][0]
                if hasattr(msg, "message") and hasattr(msg.message, "response_metadata"):
                    meta = msg.message.response_metadata
                    if "model_name" in meta:
                        model = meta["model_name"]
        except Exception:
            pass

        self._current_llm_calls.append({
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latency_ms": round(latency, 1),
        })
        global _token_total
        _token_total += prompt_tokens + completion_tokens

    def on_llm_start(self, serialized, prompts, **kwargs):
        self._llm_t_start = time.time()

    def on_tool_start(self, serialized, input_str, **kwargs):
        self._tool_t_start = time.time()
        self._tool_name = serialized.get("name", "unknown") if serialized else "unknown"

    def on_tool_end(self, output, **kwargs):
        latency = (time.time() - getattr(self, '_tool_t_start', time.time())) * 1000
        self._current_tool_calls.append({
            "tool": getattr(self, '_tool_name', 'unknown'),
            "latency_ms": round(latency, 1),
        })


# ================================================================
# Graph monitoring helpers
# ================================================================

def record_graph_run(events: list[dict], agent_name: str, state: dict, result: dict,
                     llm_calls: list[dict], tool_calls: list[dict],
                     total_latency_ms: float, error: str | None = None):
    """Write a log entry after a graph invocation (used by session.invoke for the overall run)."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "agent": agent_name,
        "input": _summarize_state(state),
        "output": _summarize_result(result),
        "llm_calls": llm_calls,
        "tool_calls": tool_calls,
        "total_latency_ms": round(total_latency_ms, 1),
        "error": error,
    }
    _session_logs.append(entry)
    _write_log(entry)


def _summarize_state(state: dict) -> dict:
    out = {}
    for k, v in state.items():
        if k == "messages":
            out[k] = f"[{len(v)} messages]"
        elif isinstance(v, str):
            out[k] = f"[str:{len(v)} chars]"
        elif isinstance(v, list):
            out[k] = f"[list:{len(v)} items]"
        elif isinstance(v, dict):
            out[k] = f"[dict:{len(json.dumps(v, ensure_ascii=False))} chars]"
        else:
            out[k] = str(v)
    return out


def _summarize_result(result: dict) -> dict:
    return _summarize_state(result)


def _write_log(entry: dict):
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"Failed to write monitor log: {e}")


# ================================================================
# HITL monitoring helpers
# ================================================================

def on_graph_paused(stage: str):
    global _current_agent
    _current_agent = f"HITL@{stage}"


def on_graph_resumed(stage: str):
    global _current_agent
    _current_agent = f"resume@{stage}"


# ================================================================
# Forward-compat stubs for _run_with_tools / invoke_with_retry
# ================================================================

def get_active_tracer():
    return None


def get_callback():
    """Create a fresh BlogGenMonitorCallback for graph.invoke config."""
    return BlogGenMonitorCallback()


def trace_llm_call(invoke_fn, messages, model="unknown"):
    """Pass-through wrapper. Token tracking is via callbacks, not this stub."""
    return invoke_fn(messages)


def trace_tool_call(tracer, tool_name, args, result, latency_ms):
    """No-op: tool calls are tracked in the callback's _tool_calls list."""
    pass


def render_monitor_sidebar():
    import streamlit as st
    summary = get_session_summary()
    agent = summary["current_agent"]
    if agent:
        st.sidebar.info(f"🔄 当前：{agent}")
    st.sidebar.metric("图调用次数", summary["agent_count"])
    st.sidebar.metric("Token 消耗", f"{summary['total_tokens']:,}")
    recent = summary["recent"]
    if recent:
        st.sidebar.markdown("**最近：**")
        for entry in reversed(recent[-3:]):
            icon = "❌" if entry.get("error") else "✅"
            llm_count = len(entry.get("llm_calls", []))
            latency = entry.get("total_latency_ms", 0)
            st.sidebar.caption(f"{icon} {entry['agent']} ({latency:.0f}ms, {llm_count} LLM)")
