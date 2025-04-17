from typing import Union, Annotated

from langgraph.graph import MessagesState

from a2a.common.types import TaskState


class AgentState(MessagesState):
    active_agent: Annotated[Union[str, None], "Le nom de l'agent courant"]
    last_agent: Annotated[Union[str, None], "Le dernière agent utilisé"]
    action: Annotated[Union[TaskState, None], "L'action suivante à réaliser'"]
