import json
import uuid
from typing import Literal

from langchain_core.callbacks import adispatch_custom_event
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
from a2a.common.types import TaskSendParams, Message, TextPart, SendTaskResponse, AgentCard, TaskState
from a2a.utils.card_resolver import A2ACardResolver
from agent_supervisor.model import AgentState
from agent_supervisor.utils import convert_a2a_task_result_to_langchain, convert_a2a_task_event_to_langchain

llm = ChatOpenAI(model="gpt-4o-mini")

assistant_cards: dict[str, AgentCard] = {}

for agent in [
    # A2ACardResolver("http://localhost:9000").get_agent_card(),
    A2ACardResolver("http://localhost:8000").get_agent_card()
]:
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


async def router(state: AgentState) -> Command[Literal["__end__", "call_agent", "default_agent"]]:
    # Si on rentre dans le router depuis un retour d'un agent
    if isinstance(state['messages'][-1], AIMessage):

        if 'task_state' in state and state['task_state'] == TaskState.INPUT_REQUIRED:
            return Command(
                goto=END,
                update={
                    "last_agent": state['active_agent']
                }
            )

        if 'task_state' in state and state['task_state'] == TaskState.WORKING:
            return Command(
                goto=END,
                update={
                    "active_agent": None,
                    "last_agent": state['active_agent']
                }
            )

        if 'task_state' in state and state['task_state'] == TaskState.COMPLETED:
            return Command(
                goto=END,
                update={
                    "active_agent": None,
                    "last_agent": state['active_agent']
                }
            )

        raise Exception(state)

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


async def call_default_agent(state: AgentState, writer: StreamWriter):

    print(":( :( :( ")
    prompt = ChatPromptTemplate.from_messages([
        ('system', """
            ## Contexte
            Vous êtes un agent conversationnel spécialisé dans la gestion des demandes clients complexes ou spécifiques. Vous êtes sollicité lorsque les autres agents disponibles ne sont pas en mesure de fournir une réponse pertinente ou satisfaisante aux besoins du client.
            
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

        """),
        MessagesPlaceholder("messages")
    ])

    res = await (prompt | llm).ainvoke(state)
    return {
        "messages": [res],
        "action": TaskState.COMPLETED
    }


async def call_remote_agent(state: AgentState, config: RunnableConfig):
    current_agent_name = state["active_agent"]
    card = assistant_cards[current_agent_name]
    agent_cli = A2AClient(agent_card=card, url=card.url)

    parts = []
    messages = state['messages']
    parts.append(TextPart(text=messages[-1].content))

    params: TaskSendParams = TaskSendParams.model_validate({
        "id": state["task_id"] if "task_id" in state and state["task_id"] is not None else str(uuid.uuid4()),
        "sessionId": config['configurable']['thread_id'],
        "message": Message(role="user", parts=parts, metadata=None),
        "pushNotification": None,
        "historyLength": None,
        "metadata": None
    })


    if card.capabilities.streaming:
        lc_messages : list[BaseMessage] = []
        last_state = None
        async for event in agent_cli.send_task_streaming(payload=params):
            last_state = event.result.status.state
            lc_messages += convert_a2a_task_event_to_langchain(event.result)


        return {
            "task_state": last_state,
            "messages": lc_messages,
            "task_id": params.id
        }

    else:

        res: SendTaskResponse = await agent_cli.send_task(payload=params)
        lc_messages = convert_a2a_task_result_to_langchain(res.result)
        return {
            "task_state": res.result.status.state,
            "messages": lc_messages,
            "task_id": params.id
        }


graph = StateGraph(AgentState)
graph.add_node("router", router)
graph.add_node("call_agent", call_remote_agent)
graph.add_node("default_agent", call_default_agent)
graph.add_edge("call_agent", "router")
graph.add_edge("default_agent", "router")
graph.set_entry_point("router")
graph.set_finish_point("router")

agent = graph.compile(checkpointer=MemorySaver(), debug=False)
