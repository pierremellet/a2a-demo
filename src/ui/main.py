import asyncio
import uuid
from typing import Optional, Any
from uuid import UUID

from dotenv import load_dotenv
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.runnables import RunnableConfig


# Charger les variables d'environnement
load_dotenv()

from agent_supervisor.agent import agent, SupervisorMessageChunk

class CustomCallbackManager(AsyncCallbackHandler):

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        pass

    async def on_custom_event(
        self,
        name: str,
        data: Any,
        *,
        run_id: uuid.UUID,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        print(f"{data['content'].replace('\\\\', '\\')}", end="")

async def main():
    # Configuration initiale
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    run_config = RunnableConfig(callbacks= [CustomCallbackManager()], **config)

    while True:
        # Lire l'entrée utilisateur
        user_msg = input("User: ")
        print("\nAI: ", end="")

        await agent.ainvoke({"messages": [
                    HumanMessage(content=user_msg)
                ]},
        config=run_config)


        # Afficher l'état de l'agent
        agent_state = agent.get_state(config)
        print(f"\n(Agent: {agent_state.values['last_agent']})\n")

if __name__ == '__main__':
    asyncio.run(main())
