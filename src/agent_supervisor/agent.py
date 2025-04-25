import asyncio
import logging
import uuid
from typing import Literal, Union

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.types import Command, StreamWriter
from pydantic import BaseModel, Field

from a2a.client.client import A2AClient
from a2a.common.types import TaskSendParams, Message, TextPart, TaskState, \
    TaskStatusUpdateEvent, TaskArtifactUpdateEvent
from a2a.utils.card_resolver import A2ACardResolver
from agent_supervisor.logger import StdoutLogger
from agent_supervisor.model import AgentState
from agent_supervisor.utils import convert_a2a_task_result_to_langchain, convert_a2a_task_events_to_langchain, \
    QueueEndEvent, MessageEvent

logger = StdoutLogger(name="MonLogger", level=logging.DEBUG)

# Initialize the LLM model
llm = ChatOpenAI(model="gpt-4o-mini")

# Load assistant cards
assistant_cards = {
    agent.name: agent
    for agent in [
        A2ACardResolver("http://localhost:9000").get_agent_card(),
        A2ACardResolver("http://localhost:8000").get_agent_card()
    ]
}


class SupervisorMessageChunk(BaseModel):
    content: str = Field(description="Message payload")


# Define the routing prompt
ROUTING_PROMPT = """
# Instructions pour l'Agent de Routage

Tu es un agent de routage.

Ta mission est de déterminer quel agent spécialisé est le plus pertinent pour traiter la conversation en fonction du dernier message du client.

## Règles de Routage

- **Agent "coach"** : Transfère la conversation si le client demande des informations sur ses comptes ou des conseils financiers.
- **Agent "handoff"** : Transfère la conversation si le client demande à contacter un conseiller ou à obtenir de l'aide.
- **Agent "default"** : Transfère la conversation si aucun des agents spécialisés ci-dessus ne peut prendre en charge la conversation.

## Instructions

Retourne uniquement le nom de l'agent pertinent pour traiter la conversation.
"""

# Create the routing prompt template
routing_prompt_template = ChatPromptTemplate.from_messages([
    ('system', ROUTING_PROMPT),
    MessagesPlaceholder("messages")
])

# Define the default agent prompt
DEFAULT_AGENT_PROMPT = """
## Contexte
Vous êtes un agent conversationnel nommé "Failover" et spécialisé dans la gestion des demandes clients complexes ou spécifiques. 
Vous êtes sollicité lorsque les autres agents disponibles ne sont pas en mesure de fournir une réponse pertinente ou satisfaisante aux besoins du client.

## Objectif
Votre mission est de comprendre en profondeur la demande du client, de fournir des réponses précises et adaptées, et de garantir une expérience client exceptionnelle.

## Instructions

1. **Accueil et Présentation**
   - Commencez par accueillir chaleureusement le client.
   - Présentez-vous brièvement et expliquez que vous êtes là pour l'aider avec sa demande spécifique.

2. **Compréhension de la Demande**
   - Posez des questions ouvertes pour bien comprendre la demande du client.
   - Reformulez la demande pour vous assurer que vous avez bien compris.

3. **Recherche d'Informations**
   - Utilisez toutes les ressources à votre disposition pour trouver les informations nécessaires.
   - Si nécessaire, consultez des experts ou des bases de données internes.

4. **Réponse au Client**
   - Fournissez une réponse claire, concise et précise.
   - Utilisez un langage simple et compréhensible.
   - Proposez des solutions alternatives si la demande initiale ne peut pas être satisfaite.

5. **Suivi et Satisfaction**
   - Demandez au client s'il est satisfait de la réponse fournie.
   - Proposez de l'aider davantage si nécessaire.
   - Remerciez le client pour sa patience et sa confiance.
"""

# Create the default agent prompt template
default_agent_prompt_template = ChatPromptTemplate.from_messages([
    ('system', DEFAULT_AGENT_PROMPT),
    MessagesPlaceholder("messages")
])


async def router(state: AgentState) -> Command[Literal["__end__", "call_agent", "default_agent"]]:
    last_message = state['messages'][-1]

    if isinstance(last_message, AIMessage):
        task_state = state.get('task_state')
        if task_state in [TaskState.INPUT_REQUIRED, TaskState.WORKING]:
            logger.info(f"Route from {state['active_agent']} to END ")
            return Command(
                goto=END,
                update={
                    "auto_routed": False,
                    "last_agent": state['active_agent'],
                    "active_agent": None if task_state != TaskState.INPUT_REQUIRED else state['active_agent']
                }
            )

        if task_state in [TaskState.COMPLETED]:

            # Eval if need another agent or END
            if "auto_routed" not in state or state["auto_routed"] != True :
                res = await (routing_prompt_template | llm).ainvoke(state)
                agent_name = res.content
                if agent_name in ["coach", "handoff"]:
                    logger.info(f"Route from humain input to agent : {agent_name} ")
                    return Command(
                        goto="call_agent",
                        update={"active_agent": agent_name, "auto_routed":True}
                    )

            return Command(
                goto=END,
                update={
                    "auto_routed": False,
                    "last_agent": state['active_agent'],
                    "active_agent": None if task_state != TaskState.INPUT_REQUIRED else state['active_agent']
                }
            )

        logger.info(f"Task state from {state['active_agent']} is not valide : {task_state} ")

        raise Exception(state)

    if isinstance(last_message, HumanMessage):
        if state.get('active_agent'):
            logger.info(f"Route from humain input to active agent : {state.get('active_agent')} ")
            return Command(goto="call_agent")

        res = await (routing_prompt_template | llm).ainvoke(state)
        agent_name = res.content

        if agent_name in ["coach", "handoff"]:
            logger.info(f"Route from humain input to agent : {agent_name} ")
            return Command(
                goto="call_agent",
                update={"active_agent": agent_name}
            )

    return Command(
        goto="default_agent",
        update={"active_agent": "default"}
    )


async def call_default_agent(state: AgentState, writer: StreamWriter):
    buffer = ""
    async for event in (default_agent_prompt_template | llm).astream(state):
        buffer += event.content
        ui_event = MessageEvent(
            content=event.content,
            id=event.id,
            agent=state["active_agent"]
        )
        writer(ui_event)

    return {
        "messages": [AIMessage(buffer)],
        "task_state": TaskState.COMPLETED
    }


async def call_remote_agent(state: AgentState, config: RunnableConfig, writer: StreamWriter):
    current_agent_name = state["active_agent"]
    card = assistant_cards[current_agent_name]
    agent_cli = A2AClient(agent_card=card, url=card.url)

    parts = [TextPart(text=state['messages'][-1].content)]
    params = TaskSendParams(
        id=state.get("task_id") or str(uuid.uuid4()),
        sessionId=config['configurable']['thread_id'],
        message=Message(role="user", parts=parts),
        pushNotification=None,
        historyLength=None,
        metadata=None
    )

    if card.capabilities.streaming:

        logger.info(f"Call remote agent {current_agent_name} in streaming mode")
        buffer: list[Union[TaskStatusUpdateEvent, TaskArtifactUpdateEvent]] = []
        queue = asyncio.Queue(maxsize=0)

        async def next_message(q):
            async for event in agent_cli.send_task_streaming(payload=params):
                await q.put(event.result)
            await q.put(QueueEndEvent())

        asyncio.create_task(next_message(queue))

        while True:
            event: Union[TaskStatusUpdateEvent, TaskArtifactUpdateEvent, QueueEndEvent] = await queue.get()
            if isinstance(event, QueueEndEvent):
                break

            buffer.append(event)
            if isinstance(event, TaskStatusUpdateEvent) and event.status.state == TaskState.WORKING:
                for part in event.status.message.parts:
                    ui_event = MessageEvent(
                        content=part.text,
                        id=event.id,
                        agent=state["active_agent"]
                    )
                    writer(ui_event)

        last_state = buffer[-1].status.state

        ai_message = convert_a2a_task_events_to_langchain(buffer)

        logger.info(f"Remote agent {current_agent_name} streaming response completed")

        return {
            "task_state": last_state,
            "messages": [ai_message],
            "task_id": params.id
        }


    logger.info(f"Call remote agent {current_agent_name} in standard mode")
    res = await agent_cli.send_task(payload=params)
    lc_messages = convert_a2a_task_result_to_langchain(res.result)
    logger.info(f"Remote agent {current_agent_name} response completed")

    return {
        "task_state": res.result.status.state,
        "messages": lc_messages,
        "task_id": params.id
    }


# Define the state graph
graph = StateGraph(AgentState)
graph.add_node("router", router)
graph.add_node("call_agent", call_remote_agent)
graph.add_node("default_agent", call_default_agent)
graph.add_edge("call_agent", "router")
graph.add_edge("default_agent", "router")
graph.set_entry_point("router")
graph.set_finish_point("router")

# Compile the agent
supervisor_agent = graph.compile(checkpointer=MemorySaver(), debug=False)
