"""Test _chat_history_messages: dict vs LangChain object handling."""
import pytest
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from src.agents.nodes import _chat_history_messages


class TestChatHistoryMessages:
    def test_empty_history(self):
        assert _chat_history_messages([]) == []

    def test_dict_messages(self):
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
            {"role": "system", "content": "be helpful"},
        ]
        result = _chat_history_messages(history)
        assert len(result) == 3
        assert isinstance(result[0], HumanMessage)
        assert result[0].content == "hello"
        assert isinstance(result[1], AIMessage)
        assert result[1].content == "hi there"
        assert isinstance(result[2], SystemMessage)
        assert result[2].content == "be helpful"

    def test_dict_human_role(self):
        result = _chat_history_messages([{"role": "human", "content": "test"}])
        assert isinstance(result[0], HumanMessage)

    def test_dict_ai_role(self):
        result = _chat_history_messages([{"role": "ai", "content": "test"}])
        assert isinstance(result[0], AIMessage)

    def test_dict_tool_message(self):
        history = [{"role": "tool", "content": "result", "tool_call_id": "abc123"}]
        result = _chat_history_messages(history)
        assert isinstance(result[0], ToolMessage)
        assert result[0].content == "result"
        assert result[0].tool_call_id == "abc123"

    def test_langchain_objects_pass_through(self):
        """LangGraph checkpointer returns LangChain objects — must pass through."""
        history = [
            HumanMessage(content="hello"),
            AIMessage(content="response"),
            SystemMessage(content="system"),
        ]
        result = _chat_history_messages(history)
        assert len(result) == 3
        assert result[0] is history[0]  # Identity preserved
        assert result[1] is history[1]
        assert result[2] is history[2]

    def test_mixed_dict_and_objects(self):
        """Checkpointer may mix restored LangChain objects with new dict messages."""
        history = [
            HumanMessage(content="restored from checkpoint"),
            {"role": "assistant", "content": "new message"},
        ]
        result = _chat_history_messages(history)
        assert len(result) == 2
        assert result[0] is history[0]  # Pass-through
        assert isinstance(result[1], AIMessage)
        assert result[1].content == "new message"

    def test_tool_message_object_pass_through(self):
        """ToolMessage from checkpointer must pass through without m.get() call."""
        tm = ToolMessage(content="result", tool_call_id="tc1")
        result = _chat_history_messages([tm])
        assert result[0] is tm

    def test_unknown_role_defaults_to_human(self):
        result = _chat_history_messages([{"role": "unknown", "content": "blah"}])
        assert isinstance(result[0], HumanMessage)

    def test_regression_humanmessage_no_get(self):
        """Regression: HumanMessage has no .get() method → must pass through, not crash."""
        msg = HumanMessage(content="Hi, I want to learn RAG")
        result = _chat_history_messages([msg])
        assert len(result) == 1
        assert result[0] is msg
        assert result[0].content == "Hi, I want to learn RAG"
