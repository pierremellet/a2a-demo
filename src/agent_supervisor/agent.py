import json
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.types import Command, StreamWriter
from pydantic import BaseModel, Field

from a2a.client.client import A2AClient
from a2a.common.types import TaskSendParams, Message, TextPart, SendTaskResponse, AgentCard, TaskState, \
    SendTaskStreamingResponse
from a2a.utils.card_resolver import A2ACardResolver
from agent_supervisor.model import AgentState

llm = ChatOpenAI(model="gpt-4o-mini")

assistant_cards: dict[str, AgentCard] = {}

for agent in [A2ACardResolver("http://localhost:9000").get_agent_card()]:
    assistant_cards[agent.name] = agent


class SupervisorMessageChunk(BaseModel):
    content: str = Field(description="Message payload")


s_prompt = f"""

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

prompt = ChatPromptTemplate.from_messages([
    (
        'system', s_prompt
    ),
    MessagesPlaceholder("messages")
])


async def call_llm(state: AgentState) -> Command[Literal["__end__", "call_agent", "default_agent"]]:
    # Si on rentre dans le router depuis un retour d'un agent
    if isinstance(state['messages'][-1], AIMessage):

        if 'action' in state and state['action'] == TaskState.INPUT_REQUIRED:
            return Command(
                goto=END,
                update={
                    "last_agent"  : state['active_agent']
                }
            )

        if 'action' in state and state['action'] == TaskState.COMPLETED:
            return Command(
                goto=END,
                update={
                    "active_agent": None,
                    "last_agent"  : state['active_agent']
                }
            )

    # Si on rentre dans le router depuis une réponse humaine
    if isinstance(state['messages'][-1], HumanMessage):

        # Si un agent est actif, on redirige directement la conversation vers cet agent
        if 'active_agent' in state and state['active_agent'] is not None:
            return Command(
                goto="call_agent"
            )

        # Si aucun agent actif, on en recherche un par LLM
        res = await (prompt | llm).ainvoke(state)

        if res.content in ["coach", "handoff"]:
            return Command(
                goto="call_agent",
                update={
                    "active_agent": res.content
                }
            )

    # Si on tombe dans un cas non prévu, on va vers l'agent par défault
    return Command(
        goto="default_agent",
        update={
            "active_agent": "default"
        }
    )


async def call_default_agent(state: AgentState, writer:StreamWriter):

    events : list[SupervisorMessageChunk] = []
    async for event in llm.astream(state['messages'][-1].content):
        message = SupervisorMessageChunk(content=event.content)
        events.append(message)
        writer(message)
    return {
        "messages" : [AIMessage("".join([x.content for x in events]))],
        "action": TaskState.COMPLETED
    }


async def call_remote_agent(state: AgentState, config: RunnableConfig, writer:StreamWriter):
    current_agent_name = state["active_agent"]
    card = assistant_cards[current_agent_name]
    agent_cli = A2AClient(agent_card=card, url=card.url)

    parts = []
    messages = state['messages']
    parts.append(TextPart(text=messages[-1].content))

    params: TaskSendParams = TaskSendParams.model_validate({
        "id": config['configurable']['thread_id'],
        "sessionId": config['configurable']['thread_id'],
        "message": Message(role="user", parts=parts, metadata=None),
        "pushNotification": None,
        "historyLength": None,
        "metadata": None
    })

    last_event = None
    async for event in agent_cli.send_task_streaming(payload=params) :
        last_event = event
        message = SupervisorMessageChunk(content="".join([p.text for p in event.result.status.message.parts]))
        if event.result.status.state == TaskState.WORKING:
            writer(message)


    return {
        "messages" : [AIMessage("".join([x.text for x in last_event.result.status.message.parts]))],
        "action": last_event.result.status.state
    }


graph = StateGraph(AgentState)
graph.add_node("call_llm", call_llm)
graph.add_node("call_agent", call_remote_agent)
graph.add_node("default_agent", call_default_agent)
graph.add_edge("call_agent", "call_llm")
graph.add_edge("default_agent", "call_llm")

graph.set_entry_point("call_llm")

agent = graph.compile(checkpointer=MemorySaver())
