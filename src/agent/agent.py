"""Legal AI Agent implementation."""

import logging
import json
import time
from typing import Any, Iterator, Optional, cast
from openai import OpenAI

from src.config import settings
from src.agent.memory import ConversationMemory, Message
from src.agent.tools import LegalSearchTool
from src.retrieval.vector_store import SearchResult

logger = logging.getLogger(__name__)

class LegalAgent:
    """An AI Agent that can chat and use tools to answer legal questions."""
    
    def __init__(
        self,
        search_tool: LegalSearchTool,
        model_name: Optional[str] = None,
        max_turns: int = 10,
    ):
        model_name = model_name or settings.llm_model
        if not settings.openrouter_api_key:
            logger.warning("openrouter_api_key is empty. Agent will fail if called.")
            
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.openrouter_api_key,
            timeout=30.0
        )
        self.model_name = model_name
        self.memory = ConversationMemory()
        self.search_tool = search_tool
        self.max_turns = max_turns
        # Legal chunks retrieved while answering the latest message
        self.last_sources: list[SearchResult] = []
        # Timing of the latest message in seconds: first_token, llm, tools, total
        self.last_timings: dict[str, float | None] = {}
        
        self._init_system_prompt()
        
    def _init_system_prompt(self):
        system_prompt = """Bạn là một chuyên gia pháp lý (Legal AI Assistant). Nhiệm vụ của bạn là trả lời các câu hỏi về pháp luật Việt Nam một cách chính xác, khách quan và dễ hiểu.
        
HƯỚNG DẪN QUAN TRỌNG:
1. LUÔN SỬ DỤNG CÔNG CỤ: Khi người dùng hỏi về quy định pháp luật, hãy gọi công cụ `search_legal_docs` để tra cứu tài liệu trước khi trả lời. Đừng tự bịa ra điều luật.
2. TRÍCH DẪN RÕ RÀNG: Bắt buộc trích dẫn cụ thể (ví dụ: Điểm a, Khoản 1, Điều X của văn bản Y) đối với mỗi luận điểm đưa ra dựa trên kết quả tra cứu.
3. HỎI LẠI KHI CẦN THIẾT: Nếu câu hỏi quá mơ hồ hoặc thiếu ngữ cảnh, hãy hỏi lại người dùng để làm rõ thay vì tìm kiếm mù quáng.
4. TỪ CHỐI NẾU KHÔNG CÓ DỮ LIỆU: Nếu đã dùng công cụ tìm kiếm mà không có tài liệu liên quan, hãy trả lời: "Xin lỗi, tôi chưa tìm thấy quy định cụ thể nào về vấn đề này trong cơ sở dữ liệu."
5. CÂU KHÔNG LIÊN QUAN PHÁP LUẬT: Với lời chào, câu cảm ơn hoặc câu hỏi ngoài lĩnh vực pháp luật (ví dụ toán, thời tiết), KHÔNG gọi công cụ. Trả lời ngắn gọn, thân thiện (nếu câu hỏi đơn giản thì có thể trả lời luôn), rồi nhắc rằng bạn chuyên hỗ trợ các câu hỏi về pháp luật Việt Nam.
"""
        self.memory.add_message(Message(role="system", content=system_prompt))
        
    def chat(self, user_input: str) -> str:
        """Process a user message and return the agent's full response."""
        return "".join(self.chat_stream(user_input))

    def chat_stream(self, user_input: str) -> Iterator[str]:
        """Process a user message and yield the response text as it is generated.

        Sources retrieved for this message are available in `last_sources` and
        timings in `last_timings` once the stream is exhausted. If the LLM call
        fails or the consumer stops early, memory is restored so the unfinished
        turn leaves no trace.
        """
        self.memory.trim(self.max_turns)
        self.last_sources = []
        snapshot = list(self.memory.messages)
        completed = False
        try:
            yield from self._run(user_input)
            completed = True
        finally:
            if not completed:
                self.memory.messages = snapshot

    def _run(self, user_input: str) -> Iterator[str]:
        """Tool-calling loop: let the LLM search up to `max_steps` times, then answer."""
        self.memory.add_message(Message(role="user", content=user_input))

        start = time.perf_counter()
        timings: dict[str, float | None] = {"first_token": None, "llm": 0.0, "tools": 0.0}
        self.last_timings = timings

        def finish() -> None:
            timings["total"] = time.perf_counter() - start
            logger.info(
                "Chat timing: first token %s, LLM %.1fs, tools %.1fs, total %.1fs",
                f"{timings['first_token']:.1f}s" if timings["first_token"] is not None else "n/a",
                timings["llm"], timings["tools"], timings["total"],
            )

        max_steps = 3
        for step in range(max_steps):
            llm_start = time.perf_counter()
            content, tool_calls = yield from self._llm_step(step, start, timings)
            timings["llm"] += time.perf_counter() - llm_start

            if not tool_calls:
                self.memory.add_message(Message(role="assistant", content=content))
                finish()
                return

            # Keep the text of a tool-calling step apart from the next step's text
            if content:
                yield "\n\n"

            self.memory.add_message(Message(
                role="assistant",
                content=content or None,
                tool_calls=[
                    {"id": tc["id"], "type": "function",
                     "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                    for tc in tool_calls
                ],
            ))

            tools_start = time.perf_counter()
            for tc in tool_calls:
                self._run_tool(tc)
            timings["tools"] += time.perf_counter() - tools_start

        fallback = "Xin lỗi, tôi đã xử lý quá lâu nhưng chưa tìm ra câu trả lời cuối cùng."
        self.memory.add_message(Message(role="assistant", content=fallback))
        yield fallback
        finish()

    def _llm_step(
        self, step: int, chat_start: float, timings: dict[str, float | None]
    ) -> Iterator[str]:
        """Call the LLM with streaming; yield text chunks, return (text, tool_calls)."""
        logger.info("Calling LLM (Step %d)...", step + 1)
        step_start = time.perf_counter()
        stream = self.client.chat.completions.create(
            messages=cast(Any, self.memory.get_context()),
            model=self.model_name,
            tools=cast(Any, [self.search_tool.get_tool_schema()]),
            tool_choice="auto",
            stream=True,
        )

        parts: list[str] = []
        calls: dict[int, dict[str, str]] = {}
        step_first_token: float | None = None
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                now = time.perf_counter()
                if step_first_token is None:
                    step_first_token = now - step_start
                if timings["first_token"] is None:
                    timings["first_token"] = now - chat_start
                parts.append(delta.content)
                yield delta.content
            # Tool calls arrive in pieces: id/name first, arguments spread over many chunks
            for tc in delta.tool_calls or []:
                slot = calls.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                if tc.id:
                    slot["id"] = tc.id
                if tc.function:
                    if tc.function.name and not slot["name"]:
                        slot["name"] = tc.function.name
                    if tc.function.arguments:
                        slot["arguments"] += tc.function.arguments

        logger.info(
            "LLM step %d: %s, total %.1fs, %d tool call(s)",
            step + 1,
            f"first token {step_first_token:.1f}s" if step_first_token is not None else "no text",
            time.perf_counter() - step_start,
            len(calls),
        )
        return "".join(parts), [calls[i] for i in sorted(calls)]

    def _run_tool(self, tool_call: dict[str, str]) -> None:
        """Execute one tool call and store its result in memory."""
        name = tool_call["name"]
        if name != "search_legal_docs":
            logger.warning("Unknown tool called: %s", name)
            tool_result = f"Error: unknown tool '{name}'"
        else:
            tool_result = self._search(tool_call["arguments"])

        self.memory.add_message(Message(
            role="tool",
            content=tool_result,
            tool_call_id=tool_call["id"],
            name=name,
        ))

    def _search(self, arguments: str) -> str:
        """Run the legal search tool from raw JSON arguments and return text for the LLM."""
        name = "search_legal_docs"
        try:
            query = json.loads(arguments).get("query", "")
            logger.info("Executing tool: %s with query: '%s'", name, query)
            t0 = time.perf_counter()
            results = self.search_tool.search(query)
            logger.info("Tool %s: %d result(s) in %.2fs", name, len(results), time.perf_counter() - t0)
            self.last_sources.extend(results)
            return self.search_tool.format_results(results)
        except Exception as e:
            logger.error("Error executing tool %s: %s", name, e)
            return f"Error: {e}"
