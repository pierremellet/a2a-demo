from typing import Optional

from langgraph.graph import MessagesState
from openai.types.chat.completion_create_params import ResponseFormat


class AgentState(MessagesState):
    structured_response: Optional[ResponseFormat]