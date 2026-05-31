"""Test _run_with_tools: timeout, tool limits, message ordering."""
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from src.agents.nodes import _run_with_tools


def _make_mock_llm(responses: list):
    """Create a mock LLM where each invoke() returns one response in sequence."""
    mocks = [MagicMock(content=t, tool_calls=[]) for t in responses]
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = mocks
    mock_llm.bind_tools.return_value = mock_llm
    mock_llm.model_name = "test-model"
    return mock_llm


def _make_mock_llm_with_tools(responses: list):
    """Each response is (content, tool_calls_list) tuple."""
    mocks = []
    for content, tool_calls in responses:
        m = MagicMock(content=content, tool_calls=tool_calls)
        mocks.append(m)
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = mocks
    mock_llm.bind_tools.return_value = mock_llm
    mock_llm.model_name = "test-model"
    return mock_llm


class TestRunWithTools:
    def test_simple_no_tool_calls(self):
        """LLM returns text directly — return it."""
        llm = _make_mock_llm(["Hello, here is the content."])
        result = _run_with_tools(llm, "sys", "user", [], agent_name="writer")
        assert result == "Hello, here is the content."

    def test_passes_system_and_user_prompts(self):
        """Verify system + user prompts reach the LLM."""
        llm = _make_mock_llm(["ok"])
        _run_with_tools(llm, "You are a writer.", "Write chapter 1.", [])
        call_args = llm.invoke.call_args[0][0]
        assert len(call_args) == 2
        assert isinstance(call_args[0], SystemMessage)
        assert call_args[0].content == "You are a writer."
        assert isinstance(call_args[1], HumanMessage)
        assert call_args[1].content == "Write chapter 1."

    def test_binds_tools(self):
        """Tool list must be bound to the LLM."""
        tools = [MagicMock(), MagicMock()]
        llm = _make_mock_llm(["ok"])
        _run_with_tools(llm, "sys", "user", tools)
        llm.bind_tools.assert_called_once_with(tools)

    def test_tool_execution_success(self):
        """LLM calls a tool → tool executed → result fed back."""
        tc = {
            "name": "query_vector_store",
            "args": {"query": "test", "top_k": 3},
            "id": "tc1",
        }
        with patch("src.tools.query_vector_store", return_value=["doc1", "doc2"]) as mock_query:
            llm = _make_mock_llm_with_tools([
                ("", [tc]),           # Round 1: LLM calls tool
                ("Final answer.", []),  # Round 2: LLM returns text
            ])
            result = _run_with_tools(llm, "sys", "user", [], agent_name="chapter_planner")
            mock_query.assert_called_once_with(query="test", top_k=3)
            assert result == "Final answer."

    def test_max_tool_calls_limit(self):
        """When tool call count exceeds max, skip and force LLM to output."""
        tc1 = {"name": "query_vector_store", "args": {}, "id": "tc1"}
        tc2 = {"name": "query_vector_store", "args": {}, "id": "tc2"}

        with patch("src.tools.query_vector_store", return_value=["data"]):
            llm = _make_mock_llm_with_tools([
                ("", [tc1, tc2]),      # LLM requests 2 tools but limit is 1
                ("Forced output.", []),  # LLM forced to output
            ])
            result = _run_with_tools(
                llm, "sys", "user", [], agent_name="writer_chapter"
            )
            assert result == "Forced output."

    def test_max_tool_calls_skips_and_sends_toolmessage_for_each(self):
        """Each skipped tool_call must get a ToolMessage (API requirement)."""
        tc1 = {"name": "query_vector_store", "args": {}, "id": "tc1"}
        tc2 = {"name": "query_vector_store", "args": {}, "id": "tc2"}

        with patch("src.tools.query_vector_store", return_value=["data"]):
            llm = _make_mock_llm_with_tools([
                ("", [tc1, tc2]),
                ("Ok.", []),
            ])
            _run_with_tools(llm, "sys", "user", [], agent_name="reviewer")

            # The second LLM call's messages must include ToolMessages for both
            second_call_msgs = llm.invoke.call_args_list[1][0][0]
            tool_msgs = [m for m in second_call_msgs if isinstance(m, ToolMessage)]
            # tc2 was skipped (limit 1, tc1 consumed the slot) → must still have ToolMessage
            tc_ids = {m.tool_call_id for m in tool_msgs}
            assert "tc1" in tc_ids
            assert "tc2" in tc_ids

    def test_tool_result_truncation(self):
        """Long tool results must be truncated to 4000 chars."""
        tc = {"name": "query_vector_store", "args": {}, "id": "tc1"}
        long_data = [{"text": "x" * 5000}]
        with patch("src.tools.query_vector_store", return_value=long_data):
            llm = _make_mock_llm_with_tools([
                ("", [tc]),
                ("Done.", []),
            ])
            _run_with_tools(llm, "sys", "user", [], agent_name="chapter_planner")
            second_call_msgs = llm.invoke.call_args_list[1][0][0]
            tool_msg = [m for m in second_call_msgs if isinstance(m, ToolMessage)][0]
            assert len(tool_msg.content) <= 4100  # 4000 + truncation notice

    def test_timeout_expired(self):
        """When cumulative time exceeds max, raise RuntimeError."""
        tc = {"name": "query_vector_store", "args": {}, "id": "tc1"}
        with patch("src.tools.query_vector_store", return_value=["data"]):
            llm = _make_mock_llm_with_tools([
                ("", [tc]),
                ("final", []),
            ])
            with patch("time.time", side_effect=[0, 400, 400, 400, 400, 400]):
                with pytest.raises(RuntimeError, match="超时"):
                    _run_with_tools(llm, "sys", "user", [], agent_name="writer_chapter")

    def test_unknown_tool_returns_error_message(self):
        """Unknown tool name → ToolMessage with error text, not crash."""
        tc = {"name": "nonexistent_tool", "args": {}, "id": "tc1"}
        llm = _make_mock_llm_with_tools([
            ("", [tc]),
            ("Done.", []),
        ])
        result = _run_with_tools(llm, "sys", "user", [], agent_name="writer")
        second_call_msgs = llm.invoke.call_args_list[1][0][0]
        tool_msg = [m for m in second_call_msgs if isinstance(m, ToolMessage)][0]
        assert "Unknown tool" in tool_msg.content
        assert result == "Done."

    def test_tool_exception_caught(self):
        """Tool execution error → ToolMessage with error, not crash."""
        tc = {"name": "tavily_search", "args": {"query": "test"}, "id": "tc1"}
        with patch("src.agents.nodes._run_with_tools", side_effect=None):
            pass
        with patch("src.tools.tavily_search", side_effect=RuntimeError("network down")):
            llm = _make_mock_llm_with_tools([
                ("", [tc]),
                ("Fallback.", []),
            ])
            result = _run_with_tools(llm, "sys", "user", [], agent_name="writer")
            second_call_msgs = llm.invoke.call_args_list[1][0][0]
            tool_msg = [m for m in second_call_msgs if isinstance(m, ToolMessage)][0]
            assert "Tool error" in tool_msg.content
            assert result == "Fallback."

    def test_uses_agent_specific_config(self):
        """Agent name must select correct limits from config."""
        llm = _make_mock_llm(["ok"])
        with patch("src.agents.nodes.MAX_TOOL_SEC_PER_AGENT", {"test_agent": 42}):
            with patch("src.agents.nodes.MAX_TOOL_ROUNDS_PER_AGENT", {"test_agent": 1}):
                with patch("src.agents.nodes.MAX_TOOL_CALLS_PER_AGENT", {"test_agent": 0}):
                    result = _run_with_tools(llm, "sys", "user", [], agent_name="test_agent")
                    assert result == "ok"

    def test_tavily_search_tool(self):
        """Tavily search tool is dispatched correctly."""
        tc = {"name": "tavily_search", "args": {"query": "RAG", "max_results": 5}, "id": "tc1"}
        with patch("src.tools.tavily_search", return_value=[{"title": "RAG intro"}]) as mock_search:
            llm = _make_mock_llm_with_tools([
                ("", [tc]),
                ("Done.", []),
            ])
            _run_with_tools(llm, "sys", "user", [], agent_name="writer")
            mock_search.assert_called_once_with(query="RAG", max_results=5)

    def test_max_rounds_limit(self):
        """After max_rounds, final fallback call forces output."""
        tc = {"name": "query_vector_store", "args": {"query": "test"}, "id": "tc1"}
        with patch("src.tools.query_vector_store", return_value=["data"]):
            # Round 0: tool call → process tools
            # Fallback (max_rounds=1 reached): LLM returns text directly
            llm = _make_mock_llm_with_tools([
                ("", [tc]),              # Round 0: has tool_calls
                ("Final forced.", []),   # Fallback: text output (no tool_calls)
            ])
            with patch("src.agents.nodes.MAX_TOOL_ROUNDS_PER_AGENT", {"test_rnd": 1}):
                with patch("time.time", side_effect=[0, 0, 0, 0, 0, 0, 0, 0]):
                    result = _run_with_tools(llm, "sys", "user", [], agent_name="test_rnd")
                    assert result == "Final forced."
