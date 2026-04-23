# agent/state.py

from typing import TypedDict, List, Optional
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """
    The complete memory of the agent across all conversation turns.

    Fields:
    - messages: Full conversation history (HumanMessage + AIMessage objects).
                LangGraph appends to this automatically. Never clear this.
    - intent: The classified intent of the MOST RECENT user message.
              Values: "greeting" | "inquiry" | "high_intent" | "unknown"
    - lead_name: Collected name of the prospective lead. None until provided.
    - lead_email: Collected email address. None until provided.
    - lead_platform: Creator platform (YouTube, Instagram, etc.). None until provided.
    - lead_captured: Boolean flag. True only after mock_lead_capture() is called.
                     Prevents duplicate tool calls if user sends extra messages.
    - awaiting_field: Which lead field the agent is currently asking for.
                      Values: "name" | "email" | "platform" | None
                      Controls the lead collection flow precisely.
    - rag_context: The retrieved knowledge base context for the current turn.
                   Cleared after each response to avoid context overflow.
    """
    messages: List[BaseMessage]
    intent: Optional[str]
    lead_name: Optional[str]
    lead_email: Optional[str]
    lead_platform: Optional[str]
    lead_captured: bool
    awaiting_field: Optional[str]
    rag_context: Optional[str]
