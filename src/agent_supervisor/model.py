from typing import Union, Annotated

from langgraph.graph import MessagesState
from pydantic import Field, BaseModel


class AgentResponse(BaseModel):
    message: str = Field(description="La réponse")
    agent: Union[str, None] = Field(description="Le nom de l'agent à utiliser", default=None)
    agent_request: Union[str, None] = Field(description="La demande à faire à l'agent", default=None)


class AgentState(MessagesState):
    active_agent: Annotated[Union[str, None], "Le nom de l'agent courant"]
    action: Annotated[Union[str, None], "L'action suivante à réaliser'"]
