"""Conversation memory management for the AI Agent."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel

class Message(BaseModel):
    """Represents a single message in the conversation."""
    role: str  # 'system', 'user', 'assistant', 'tool'
    content: Optional[str] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None  # Used for tool name
    tool_calls: Optional[List[Dict[str, Any]]] = None

class ConversationMemory(BaseModel):
    """Manages the history of messages in a conversation."""
    messages: List[Message] = []
    
    def add_message(self, message: Message) -> None:
        """Add a new message to the memory."""
        self.messages.append(message)
        
    def get_context(self) -> List[Dict[str, Any]]:
        """Return the messages in a format suitable for the LLM API."""
        formatted = []
        for msg in self.messages:
            m: Dict[str, Any] = {"role": msg.role}
            if msg.content is not None:
                m["content"] = msg.content
            if msg.name:
                m["name"] = msg.name
            if msg.tool_call_id:
                m["tool_call_id"] = msg.tool_call_id
            if msg.tool_calls:
                m["tool_calls"] = msg.tool_calls
            formatted.append(m)
        return formatted
    
    def clear(self) -> None:
        """Clear the conversation memory."""
        self.messages = []

    def trim(self, max_turns: int) -> None:
        """Keep system messages plus the last `max_turns` user turns.

        A turn starts at a user message and includes the assistant/tool messages
        after it, so tool results always stay next to the tool call that made them.
        """
        user_idx = [i for i, m in enumerate(self.messages) if m.role == "user"]
        if len(user_idx) <= max_turns:
            return
        cut = user_idx[-max_turns]
        system = [m for m in self.messages[:cut] if m.role == "system"]
        self.messages = system + self.messages[cut:]
