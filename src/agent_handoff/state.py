from typing import Optional, Union

from langgraph.graph import MessagesState
from openai.types.chat.completion_create_params import ResponseFormat


class AgentState(MessagesState):
    structured_response: Optional[Union[ResponseFormat, None]]