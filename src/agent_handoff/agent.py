from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph

from a2a.server.base_agent import BaseAgent
from a2a.common.common import ResponseFormat
from a2a.common.types import AgentSkill, AgentCard, AgentCapabilities
from agent_handoff.state import AgentState


class HandoffAgent(BaseAgent):
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    def __init__(self):
        super().__init__()
        llm = ChatOpenAI(model="gpt-4o-mini")

        prompt = ChatPromptTemplate.from_messages([
            (
                'system',
                """
                Tu es en charge d'accompagner un client pour prendre contact avec un conseiller clientèle.
                                         
                # Protocol de prise de prise de contact :
                1. Commence par demander au client la raison de son contact
                2. Demande au client le canal de contact qu'il souhaite entre email et téléphone
                3. Selon le canal de contact sélectionné par le client applique les règles suivantes :
                    - Si l'humain veut prendre contact par mail, propose de l'aide pour rédigier l'email.
                    - Si l'humain veut prendre contact par téléphone, demande lui quand il est joignable pour être rappeler. Demande dans le cas d'un contact par téléphone le numéro de téléphone de l'humain pour le rappeler.
                
                """
            ),
            MessagesPlaceholder("messages")
        ])

        def call_llm(state: AgentState):
            res: ResponseFormat = (prompt | llm.with_structured_output(ResponseFormat)).invoke(state)
            return {
                "messages": [AIMessage(res.message)],
                "structured_response": res
            }

        graph = StateGraph(AgentState)

        graph.add_node("call_llm", call_llm)

        graph.set_entry_point("call_llm")
        graph.set_finish_point("call_llm")

        self.set_agent(graph.compile(checkpointer=MemorySaver()))

    def get_agent_capabilities(self) -> AgentCapabilities:
        return AgentCapabilities(streaming=True, pushNotifications=True)

    def get_agent_card(self) -> AgentCard:
        return AgentCard(
            name="handoff",
            description="Permet d'accompagner un client dans la prise de contact vers un conseiller",
            url=f"http://localhost:9000",
            version="1.0.0",
            defaultInputModes=HandoffAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=HandoffAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=self.get_agent_capabilities(),
            skills=self.get_agent_skills(),
        )

    def get_agent_skills(self) -> list[AgentSkill]:
        return [AgentSkill(
            id="handoff",
            name="Agent Handoff",
            description="Capacité d'identification du motif de contact",
            tags=["handoff", "mail", "phone"],
            examples=["Comment contacter un conseiller ?"],
        )
        ]
