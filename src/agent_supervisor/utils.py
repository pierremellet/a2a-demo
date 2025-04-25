from typing import Union, TypedDict

from langchain_core.messages import AIMessage, AIMessageChunk

from a2a.common.types import Task, TaskArtifactUpdateEvent, TaskStatusUpdateEvent

class QueueEndEvent:
    pass


class MessageEvent(TypedDict):
    content : str
    id : str
    agent : str


def convert_a2a_task_events_to_langchain(events :  list[Union[TaskStatusUpdateEvent, TaskArtifactUpdateEvent]]) -> AIMessage:

    content = ""

    for event in events :
        if isinstance(event, TaskStatusUpdateEvent):
            if "message" in event.status :
                if "parts" in event.status.message and event.status.message.parts is not None:
                    for part in event.status.message.parts:
                        content += part.text


        if isinstance(event, TaskArtifactUpdateEvent):
            if event.artifact is not None :
                if event.artifact.parts is not None:
                    for part in event.artifact.parts:
                        content += part.text

    return AIMessage(content=content)


def convert_a2a_task_result_to_langchain(task: Task) -> list[AIMessage]:
    messages : list[AIMessage] = []


    if "parts" in task.status.message and task.status.message.parts is not None:
        for part in task.status.message.parts:
            messages.append(AIMessage(content=part.text, response_metadatas=part.metadata))

    if task.artifacts is not None:
        for artifact in task.artifacts:
            for part in artifact.parts:
                messages.append(AIMessage(content=part.text, response_metadatas=part.metadata))

    return messages