"""Unit tests for agent memory and the chat loop (fake LLM client, no network)."""

import json
from types import SimpleNamespace

import pytest

from src.agent.agent import LegalAgent
from src.agent.memory import ConversationMemory, Message
from src.agent.tools import LegalSearchTool
from src.retrieval.vector_store import SearchResult


def make_result(chunk_id: str) -> SearchResult:
    """Build a minimal SearchResult."""
    return SearchResult(
        chunk_id=chunk_id,
        doc_id=39,
        article_id="100/2019/nđ-cp#6",
        article_number=6,
        title="Văn bản số 100/2019/NĐ-CP",
        category="nđ-cp",
        text="Phạt tiền từ 200.000 đồng đến 300.000 đồng",
        score=0.5,
        clause_number=2,
    )


class FakeRetriever:
    """Retriever stub returning fixed results."""

    def __init__(self, results: list[SearchResult]):
        self.results = results
        self.queries: list[str] = []

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        self.queries.append(query)
        return self.results


def text_reply(content: str) -> SimpleNamespace:
    """LLM response with a plain answer."""
    message = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def tool_reply(query: str) -> SimpleNamespace:
    """LLM response asking to call search_legal_docs."""
    call = SimpleNamespace(
        id="call_1",
        type="function",
        function=SimpleNamespace(name="search_legal_docs", arguments=json.dumps({"query": query})),
    )
    message = SimpleNamespace(content=None, tool_calls=[call])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeCompletions:
    """Returns queued responses in order; an Exception in the queue is raised."""

    def __init__(self, replies: list):
        self.replies = list(replies)
        self.calls: list[list[dict]] = []

    def create(self, messages, **kwargs):
        self.calls.append(messages)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def make_agent(replies: list, results: list[SearchResult] | None = None, max_turns: int = 10):
    """Agent wired to a fake LLM and fake retriever."""
    retriever = FakeRetriever(results or [])
    agent = LegalAgent(LegalSearchTool(retriever), model_name="fake", max_turns=max_turns)  # type: ignore[arg-type]
    completions = FakeCompletions(replies)
    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))  # type: ignore[assignment]
    return agent, completions, retriever


def test_memory_trim_keeps_system_and_last_turns():
    memory = ConversationMemory()
    memory.add_message(Message(role="system", content="sys"))
    for i in range(3):
        memory.add_message(Message(role="user", content=f"q{i}"))
        memory.add_message(Message(role="assistant", content=f"a{i}"))
    memory.trim(max_turns=2)
    assert [m.content for m in memory.messages] == ["sys", "q1", "a1", "q2", "a2"]


def test_memory_trim_keeps_tool_messages_with_their_turn():
    memory = ConversationMemory()
    memory.add_message(Message(role="system", content="sys"))
    memory.add_message(Message(role="user", content="q0"))
    memory.add_message(Message(role="user", content="q1"))
    memory.add_message(Message(role="assistant", tool_calls=[{"id": "c"}]))
    memory.add_message(Message(role="tool", content="kết quả", tool_call_id="c"))
    memory.add_message(Message(role="assistant", content="a1"))
    memory.trim(max_turns=1)
    assert [m.role for m in memory.messages] == ["system", "user", "assistant", "tool", "assistant"]


def test_memory_trim_noop_when_short():
    memory = ConversationMemory()
    memory.add_message(Message(role="user", content="q"))
    memory.trim(max_turns=5)
    assert len(memory.messages) == 1


def test_chat_without_tool_has_no_sources():
    agent, completions, retriever = make_agent([text_reply("Xin chào!")])
    assert agent.chat("hello") == "Xin chào!"
    assert agent.last_sources == []
    assert retriever.queries == []


def test_chat_with_tool_collects_sources():
    agent, completions, retriever = make_agent(
        [tool_reply("mũ bảo hiểm"), text_reply("Phạt 200.000 - 300.000 đồng")],
        results=[make_result("39_1")],
    )
    assert agent.chat("Không đội mũ bảo hiểm phạt bao nhiêu?") == "Phạt 200.000 - 300.000 đồng"
    assert retriever.queries == ["mũ bảo hiểm"]
    assert [r.chunk_id for r in agent.last_sources] == ["39_1"]
    # Second LLM call sees the tool result with the clause citation
    tool_msgs = [m for m in completions.calls[1] if m["role"] == "tool"]
    assert "Điều 6, Khoản 2" in tool_msgs[0]["content"]


def test_sources_reset_each_message():
    agent, _, _ = make_agent(
        [tool_reply("q"), text_reply("a1"), text_reply("a2")],
        results=[make_result("39_1")],
    )
    agent.chat("câu 1")
    agent.chat("cảm ơn")
    assert agent.last_sources == []


def test_chat_remembers_previous_turns():
    agent, completions, _ = make_agent([text_reply("a1"), text_reply("a2")])
    agent.chat("Xe máy không đội mũ bảo hiểm?")
    agent.chat("Thế còn xe đạp điện?")
    contents = [m.get("content") for m in completions.calls[1]]
    assert "Xe máy không đội mũ bảo hiểm?" in contents
    assert "a1" in contents


def test_failed_chat_restores_memory():
    agent, _, _ = make_agent([RuntimeError("rate limited")])
    before = list(agent.memory.messages)
    with pytest.raises(RuntimeError):
        agent.chat("hello")
    assert agent.memory.messages == before


def test_chat_trims_memory():
    agent, _, _ = make_agent([text_reply(f"a{i}") for i in range(4)], max_turns=2)
    for i in range(4):
        agent.chat(f"q{i}")
    users = [m.content for m in agent.memory.messages if m.role == "user"]
    # Trim runs before each new message: the last max_turns previous turns + the current one
    assert users == ["q1", "q2", "q3"]
    assert agent.memory.messages[0].role == "system"
