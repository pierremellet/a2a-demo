from typing import Union

from langchain_core.messages import AIMessage, AIMessageChunk

from a2a.common.types import Task, TaskArtifactUpdateEvent, TaskStatusUpdateEvent


def convert_a2a_task_event_to_langchain(event : Union[TaskArtifactUpdateEvent, TaskStatusUpdateEvent]) -> list[AIMessageChunk]:
    messages : list[AIMessageChunk] = []

    if isinstance(event, TaskStatusUpdateEvent):
        if event.status.message.parts is not None:
            for part in event.status.message.parts:
                msg = AIMessageChunk(content = part.text, response_metadata={
                    "state" : event.status.state,
                    "a2a_meta" : event.metadata
                })
                messages.append(msg)


    if isinstance(event, TaskArtifactUpdateEvent):
        if event.artifact is not None :
            if event.artifact.parts is not None:
                for part in event.artifact.parts:
                    msg = AIMessageChunk(content=part.text, response_metadata={
                        "state": event.status.state,
                        "a2a_meta": event.metadata
                    })
                    messages.append(msg)

    return messages


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