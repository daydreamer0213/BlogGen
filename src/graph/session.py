"""Session manager: Streamlit ↔ LangGraph bridge.

One BlogGenSession per browser tab (stored in st.session_state).
Uses MemorySaver for checkpointing — state persists within a browser tab lifetime
but is lost on page refresh. Thread ID is a UUID generated at session creation.

Key contract:
  - create() → init or resume a session with a thread_id
  - invoke() → run the graph from current state to next interrupt (or end)
  - get_state() → read current state snapshot without advancing
  - update_state() → mutate state (for HITL approval, user edits)
  - is_interrupted() → check if graph is waiting at a HITL checkpoint
"""
import copy
import uuid
import time
from src.graph.state import initial_state
from src.graph.builder import compile_graph
from src import monitor


class BlogGenSession:
    """Manages a single BlogGen workflow session with native graph execution."""

    HITL_NODES = ["needs_alignment", "knowledge_tree", "chapter_planner", "review_batch"]

    def __init__(self, interrupt_after: list[str] | None = None):
        self._graph = None
        self.thread_id: str = ""
        self.config: dict = {}
        self._interrupt_after = interrupt_after if interrupt_after is not None else self.HITL_NODES

    @property
    def graph(self):
        if self._graph is None:
            self._graph = compile_graph(interrupt_after=self._interrupt_after)
        return self._graph

    def create(self, thread_id: str | None = None) -> dict:
        if thread_id:
            self.thread_id = thread_id
            self.config = {"configurable": {"thread_id": thread_id}}
            existing = self.get_state()
            if existing and existing.get("stage", ""):
                return existing
        else:
            self.thread_id = str(uuid.uuid4())
            self.config = {"configurable": {"thread_id": self.thread_id}}

        state = initial_state()
        self.graph.update_state(self.config, state)
        return state

    def get_state(self) -> dict:
        if not self.thread_id:
            return initial_state()
        snap = self.graph.get_state(self.config)
        return snap.values if snap.values is not None else initial_state()

    def update_state(self, updates: dict) -> None:
        if not self.thread_id:
            self.create()
        self.graph.update_state(self.config, updates)

    def invoke(self) -> dict:
        """Run graph + log per-node timing via LangChain callbacks."""
        cb = monitor.BlogGenMonitorCallback()
        config_with_cb = copy.deepcopy(self.config)
        config_with_cb.setdefault("callbacks", [])
        if isinstance(config_with_cb["callbacks"], list):
            config_with_cb["callbacks"].append(cb)

        state_before = self.get_state()
        t_start = time.time()
        error = None

        try:
            result = self.graph.invoke(None, config_with_cb)
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            result = dict(state_before)
            result["_error"] = str(e)

        latency = (time.time() - t_start) * 1000
        stage = result.get("stage", "error")

        # Write summary log entry
        # Collect all LLM calls from per-node entries
        all_llm_calls = []
        all_tool_calls = []
        for entry in getattr(cb, '_node_entries', []):
            all_llm_calls.extend(entry.get("llm_calls", []))
            all_tool_calls.extend(entry.get("tool_calls", []))
        # Fallback: use flat accumulators if no per-node entries
        if not all_llm_calls:
            all_llm_calls = getattr(cb, '_current_llm_calls', [])
        if not all_tool_calls:
            all_tool_calls = getattr(cb, '_current_tool_calls', [])

        monitor.record_graph_run(
            events=[], agent_name=_agent_name_from_stage(stage),
            state=state_before, result=result,
            llm_calls=all_llm_calls, tool_calls=all_tool_calls,
            total_latency_ms=latency, error=error,
        )
        if self.is_interrupted():
            monitor.on_graph_paused(stage)
        return result


    def is_interrupted(self) -> bool:
        snap = self.graph.get_state(self.config)
        if snap and snap.next and snap.next != ():
            return len(snap.next) > 0
        return False


def _agent_name_from_stage(stage: str) -> str:
    return {
        "needs_alignment": "NeedsAlignment",
        "needs_alignment_done": "NeedsAlignment",
        "knowledge_tree_done": "KnowledgeTreeBuilder",
        "chapter_plan_done": "ChapterPlanner",
        "writer_done": "Writer",
        "review_pass": "Reviewer",
        "review_reject": "Reviewer",
    }.get(stage, stage)
