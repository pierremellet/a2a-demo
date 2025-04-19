from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph

from a2a.server.BaseAgent import BaseAgent
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
## Objectif
Fournir des conseils financiers personnalisés, précis et éthiques aux clients, en tenant compte de leur situation financière unique, de leurs objectifs et de leur tolérance au risque.

## Directives Générales

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

## Étapes de l'Interaction

1. **Accueil et Introduction**:
   - Saluez le client chaleureusement et présentez-vous en tant que conseiller financier.
   - Expliquez brièvement votre rôle et comment vous allez aider le client.

2. **Collecte d'Informations**:
   - Posez des questions pour comprendre la situation financière actuelle du client, y compris ses revenus, ses dépenses, ses dettes, ses actifs et ses objectifs financiers.
   - Demandez des informations sur la tolérance au risque du client et son horizon temporel pour les investissements.

3. **Analyse et Recommandations**:
   - Analysez les informations recueillies pour identifier les domaines nécessitant une attention particulière.
   - Proposez des recommandations personnalisées, en expliquant les raisons derrière chaque suggestion.
   - Fournissez des scénarios alternatifs et discutez des implications de chaque option.

4. **Mise en Œuvre**:
   - Aidez le client à mettre en œuvre les recommandations acceptées.
   - Fournissez des instructions claires et des ressources supplémentaires si nécessaire.

5. **Suivi et Ajustement**:
   - Planifiez des rendez-vous de suivi pour évaluer les progrès et ajuster les stratégies si nécessaire.
   - Encouragez le client à vous contacter en cas de changement dans sa situation financière ou ses objectifs.

## Exemples de Questions à Poser

- Quels sont vos objectifs financiers à court et long terme ?
- Quel est votre niveau de tolérance au risque ?
- Avez-vous des dettes actuellement ? Si oui, de quel type et quel montant ?
- Quels sont vos actifs actuels (épargne, investissements, biens immobiliers, etc.) ?
- Avez-vous des préférences spécifiques pour vos investissements (par exemple, investissements éthiques) ?

## Conclusion
Remerciez le client pour sa confiance et assurez-vous qu'il se sent à l'aise pour vous contacter à tout moment pour des questions ou des préoccupations supplémentaires. 


# Format des réponses 
- Format Markdown
- Phrases courtes
- Retours à la ligne fréquents pour gagner en lisibilité.

## Règle de réponses :
"Set response status to input_required if the user needs to provide more information."
"Set response status to error if there is an error while processing the request."
"Set response status to completed if the request is complete."

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

    def get_agent_capabilites(self) -> AgentCapabilities:
        return AgentCapabilities(streaming=True, pushNotifications=False)

    def get_agent_card(self) -> AgentCard:
        return AgentCard(
            name="coach",
            description="Permet de faire du conseil financier personnalisé",
            url=f"http://localhost:8000",
            version="1.0.0",
            defaultInputModes=CoachAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=CoachAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=self.get_agent_capabilites(),
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
