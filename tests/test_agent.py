"""Unit tests for agent memory and the chat loop (fake LLM client, no network)."""

import json
from types import SimpleNamespace

import pytest

from src.agent.agent import LegalAgent
from src.agent.memory import ConversationMemory, Message
from src.agent.tools import LegalSearchTool, SearchError
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


def chunk(content: str | None = None, tool_calls: list | None = None) -> SimpleNamespace:
    """One streaming chunk, shaped like an OpenAI chat.completion.chunk."""
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


def text_reply(content: str) -> list[SimpleNamespace]:
    """Streamed plain answer, split in two chunks."""
    mid = len(content) // 2
    return [chunk(content[:mid]), chunk(content[mid:])]


def tool_reply(query: str) -> list[SimpleNamespace]:
    """Streamed request to call search_legal_docs; arguments arrive in pieces."""
    arguments = json.dumps({"query": query})
    mid = len(arguments) // 2
    first = SimpleNamespace(
        index=0, id="call_1", function=SimpleNamespace(name="search_legal_docs", arguments=arguments[:mid])
    )
    rest = SimpleNamespace(index=0, id=None, function=SimpleNamespace(name=None, arguments=arguments[mid:]))
    return [chunk(tool_calls=[first]), chunk(tool_calls=[rest])]


class FakeCompletions:
    """Returns queued chunk lists in order as streams; an Exception in the queue is raised."""

    def __init__(self, replies: list):
        self.replies = list(replies)
        self.calls: list[list[dict]] = []

    def create(self, messages, **kwargs):
        self.calls.append(messages)
        assert kwargs.get("stream") is True
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return iter(reply)


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


def test_chat_stream_yields_chunks_in_order():
    agent, _, _ = make_agent([text_reply("Xin chào bạn!")])
    chunks = list(agent.chat_stream("hello"))
    assert len(chunks) == 2
    assert "".join(chunks) == "Xin chào bạn!"
    assert agent.memory.messages[-1].content == "Xin chào bạn!"


def test_chat_stream_with_tool_streams_only_the_answer():
    agent, _, retriever = make_agent(
        [tool_reply("mũ bảo hiểm"), text_reply("Phạt 200.000 đồng")],
        results=[make_result("39_1")],
    )
    assert "".join(agent.chat_stream("Mũ bảo hiểm?")) == "Phạt 200.000 đồng"
    assert retriever.queries == ["mũ bảo hiểm"]  # arguments split across chunks were joined
    assert [r.chunk_id for r in agent.last_sources] == ["39_1"]


def test_chat_stream_records_timings():
    agent, _, _ = make_agent([tool_reply("q"), text_reply("trả lời")], results=[make_result("39_1")])
    list(agent.chat_stream("hỏi"))
    t = agent.last_timings
    assert t["first_token"] is not None and 0 <= t["first_token"] <= t["total"]
    assert t["llm"] >= 0 and t["tools"] >= 0


def test_chat_stream_closed_early_restores_memory():
    agent, _, _ = make_agent([text_reply("một câu trả lời dài")])
    before = list(agent.memory.messages)
    stream = agent.chat_stream("hello")
    next(stream)
    stream.close()  # e.g. the HTTP client disconnected
    assert agent.memory.messages == before


def test_unknown_tool_gets_an_error_result_in_memory():
    call = SimpleNamespace(index=0, id="call_x", function=SimpleNamespace(name="no_such_tool", arguments="{}"))
    agent, completions, _ = make_agent([[chunk(tool_calls=[call])], text_reply("xong")])
    assert "".join(agent.chat_stream("hi")) == "xong"
    tool_msgs = [m for m in completions.calls[1] if m["role"] == "tool"]
    assert tool_msgs[0]["tool_call_id"] == "call_x" and "unknown tool" in tool_msgs[0]["content"]


def test_search_failure_stops_the_chat_and_restores_memory():
    """A failing search must not turn into an answer written without sources."""
    class BrokenTool(LegalSearchTool):
        def search(self, query):
            raise SearchError("qdrant down")

    agent, completions, _ = make_agent([tool_reply("mũ bảo hiểm"), text_reply("câu trả lời không căn cứ")])
    agent.search_tool = BrokenTool(FakeRetriever([]))  # type: ignore[arg-type]
    before = list(agent.memory.messages)

    with pytest.raises(SearchError):
        agent.chat("Không đội mũ bảo hiểm phạt bao nhiêu?")

    assert len(completions.calls) == 1  # no second LLM call after the failure
    assert agent.memory.messages == before


def test_bad_tool_arguments_go_back_to_the_llm():
    call = SimpleNamespace(index=0, id="c1", function=SimpleNamespace(name="search_legal_docs", arguments="{not json"))
    agent, completions, retriever = make_agent([[chunk(tool_calls=[call])], text_reply("xin lỗi")])
    assert "".join(agent.chat_stream("hi")) == "xin lỗi"
    tool_msgs = [m for m in completions.calls[1] if m["role"] == "tool"]
    assert "invalid arguments" in tool_msgs[0]["content"]
    assert retriever.queries == []
