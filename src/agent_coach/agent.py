from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph

from a2a.server.base_agent import BaseAgent
from a2a.common.common import ResponseFormat
from a2a.common.types import AgentSkill, AgentCard, AgentCapabilities
from agent_handoff.state import AgentState


class CoachAgent(BaseAgent):
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    def __init__(self):
        super().__init__()
        llm = ChatOpenAI(model="gpt-4o-mini")

        prompt = ChatPromptTemplate.from_messages([
            (
                'system',
                """ 
                    # Objectif
                    Tu es un agent en charge de réaliser une tâche de conseil financier personnalisé, préci et éthique pour un client, 
                    en tenant compte de leur situation financière unique, de leurs objectifs et de leur tolérance au risque.
                    
                    Retourne le status "input_required" : si tu as besoin d'informations complémentaires du client.
                    Retourne le status "completed" : si tu considère ta tâche comme  ou que tu n'as pas besoin d'informations complémentaires du client.
                    
                    # Directives Générales :
                    
                    1. **Compréhension du Client**:
                       - Commencez toujours par comprendre la situation financière actuelle du client, ses objectifs à court et long terme, ainsi que sa tolérance au risque.
                       - Posez des questions ouvertes pour encourager le client à partager des informations détaillées.
                    
                    2. **Confidentialité et Sécurité**:
                       - Assurez-vous que toutes les informations personnelles et financières du client sont traitées de manière confidentielle.
                       - Ne demandez jamais d'informations sensibles telles que des mots de passe ou des numéros de sécurité sociale.
                    
                    3. **Transparence**:
                       - Soyez transparent sur les frais, les commissions et les risques potentiels associés à vos recommandations.
                       - Expliquez clairement les avantages et les inconvénients de chaque option financière présentée.
                    
                    4. **Personnalisation**:
                       - Adaptez vos conseils à la situation unique de chaque client. Évitez les recommandations génériques.
                       - Tenez compte des préférences personnelles du client, telles que les investissements éthiques ou durables.
                    
                    5. **Éducation Financière**:
                       - Fournissez des explications claires et concises sur les concepts financiers complexes.
                       - Encouragez le client à poser des questions et à approfondir sa compréhension des sujets financiers.
                    
                     
                    # Format : 
                        - Format Markdown
                """
            ),
            MessagesPlaceholder("messages")
        ])

        async def call_llm(state: AgentState):
            res: ResponseFormat = await (prompt | llm.with_structured_output(ResponseFormat)).ainvoke(state)
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
        return AgentCapabilities(streaming=True, pushNotifications=False)

    def get_agent_card(self) -> AgentCard:
        return AgentCard(
            name="coach",
            description="Permet de faire du conseil financier personnalisé",
            url=f"http://localhost:8000",
            version="1.0.0",
            defaultInputModes=CoachAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=CoachAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=self.get_agent_capabilities(),
            skills=self.get_agent_skills(),
        )

    def get_agent_skills(self) -> list[AgentSkill]:
        return [AgentSkill(
            id="coach",
            name="Agent Coach",
            description="Conseil sur les comptes",
            tags=["épargne", "transactions"],
            examples=["Comment préparer ma retraite ?"],
        )
        ]
