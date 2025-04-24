from typing import Union, Annotated

from langgraph.graph import MessagesState

from a2a.common.types import TaskState


class AgentState(MessagesState):
    active_agent: Annotated[Union[str, None], "Le nom de l'agent courant"]
    task_id:  Annotated[Union[str, None], "L'identifiant de la tâche A2A en cours"]
    task_state:  Annotated[Union[TaskState, None], "Last task state"]
    last_agent: Annotated[Union[str, None], "Le dernière agent utilisé"]
