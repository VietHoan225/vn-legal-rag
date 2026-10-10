"""Legal AI Agent implementation."""

import logging
import json
from typing import Optional, Any, cast
from openai import OpenAI

from src.config import settings
from src.agent.memory import ConversationMemory, Message
from src.agent.tools import LegalSearchTool

logger = logging.getLogger(__name__)

class LegalAgent:
    """An AI Agent that can chat and use tools to answer legal questions."""
    
    def __init__(self, search_tool: LegalSearchTool, model_name: Optional[str] = None):
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
        
        self._init_system_prompt()
        
    def _init_system_prompt(self):
        system_prompt = """Bạn là một chuyên gia pháp lý (Legal AI Assistant). Nhiệm vụ của bạn là trả lời các câu hỏi về pháp luật Việt Nam một cách chính xác, khách quan và dễ hiểu.
        
HƯỚNG DẪN QUAN TRỌNG:
1. LUÔN SỬ DỤNG CÔNG CỤ: Khi người dùng hỏi về quy định pháp luật, hãy gọi công cụ `search_legal_docs` để tra cứu tài liệu trước khi trả lời. Đừng tự bịa ra điều luật.
2. TRÍCH DẪN RÕ RÀNG: Bắt buộc trích dẫn cụ thể (ví dụ: Điểm a, Khoản 1, Điều X của văn bản Y) đối với mỗi luận điểm đưa ra dựa trên kết quả tra cứu.
3. HỎI LẠI KHI CẦN THIẾT: Nếu câu hỏi quá mơ hồ hoặc thiếu ngữ cảnh, hãy hỏi lại người dùng để làm rõ thay vì tìm kiếm mù quáng.
4. TỪ CHỐI NẾU KHÔNG CÓ DỮ LIỆU: Nếu đã dùng công cụ tìm kiếm mà không có tài liệu liên quan, hãy trả lời: "Xin lỗi, tôi chưa tìm thấy quy định cụ thể nào về vấn đề này trong cơ sở dữ liệu."
"""
        self.memory.add_message(Message(role="system", content=system_prompt))
        
    def chat(self, user_input: str) -> str:
        """Process a user message and return the agent's response."""
        self.memory.add_message(Message(role="user", content=user_input))
        
        max_steps = 3
        for step in range(max_steps):
            messages = self.memory.get_context()
            tools = [self.search_tool.get_tool_schema()]
            
            logger.info("Calling LLM (Step %d)...", step + 1)
            response = self.client.chat.completions.create(
                messages=cast(Any, messages),
                model=self.model_name,
                tools=cast(Any, tools),
                tool_choice="auto"
            )
            
            response_message = response.choices[0].message
            
            if not response_message.tool_calls:
                # Trả lời bình thường
                final_answer = response_message.content or ""
                self.memory.add_message(Message(role="assistant", content=final_answer))
                return final_answer
                
            # LLM quyết định gọi tool
            tool_calls_dict = []
            for tc in response_message.tool_calls:
                if tc.type == "function":
                    func_name = tc.function.name
                    func_args = tc.function.arguments
                elif getattr(tc, "type", None) == "custom":
                    func_name = tc.custom.name
                    func_args = tc.custom.input
                else:
                    logger.warning("Unknown tool call type: %s", getattr(tc, "type", "unknown"))
                    continue
                    
                tool_calls_dict.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": func_name,
                        "arguments": func_args
                    }
                })
                
            # Thêm tin nhắn của assistant (yêu cầu gọi tool) vào memory
            self.memory.add_message(Message(
                role="assistant",
                content=response_message.content,
                tool_calls=tool_calls_dict
            ))
            
            # Thực thi các tool calls
            for tool_call in response_message.tool_calls:
                if tool_call.type == "function":
                    function_name = tool_call.function.name
                    function_args = tool_call.function.arguments
                elif getattr(tool_call, "type", None) == "custom":
                    function_name = tool_call.custom.name
                    function_args = tool_call.custom.input
                else:
                    continue
                
                if function_name == "search_legal_docs":
                    try:
                        kwargs = json.loads(function_args)
                        query = kwargs.get("query", "")
                        logger.info("Executing tool: %s with query: '%s'", function_name, query)
                        
                        tool_result = self.search_tool.execute(query)
                    except Exception as e:
                        logger.error("Error executing tool %s: %s", function_name, e)
                        tool_result = f"Error: {e}"
                        
                    # Lưu kết quả của tool vào memory
                    self.memory.add_message(Message(
                        role="tool",
                        content=tool_result,
                        tool_call_id=tool_call.id,
                        name=function_name
                    ))
                else:
                    logger.warning("Unknown tool called: %s", function_name)
                    
        # Nếu vượt quá max_steps
        fallback = "Xin lỗi, tôi đã xử lý quá lâu nhưng chưa tìm ra câu trả lời cuối cùng."
        self.memory.add_message(Message(role="assistant", content=fallback))
        return fallback
