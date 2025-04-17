import asyncio
import uuid
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from agent_supervisor.agent import agent

# Charger les variables d'environnement
load_dotenv()

async def main():
    # Configuration initiale
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    run_config = RunnableConfig(callbacks=[], **config)

    while True:
        # Lire l'entrée utilisateur
        user_msg = input("User: ")
        print("\nAI: ", end="")

        # Stream des messages de l'agent
        async for msg in agent.astream(
            {
                "messages": [
                    HumanMessage(content=user_msg)
                ]
            },
            stream_mode="custom",
            config=run_config
        ):
            print(msg.content, end="")

        # Afficher l'état de l'agent
        agent_state = agent.get_state(config)
        print(f"\n\tAgent: {agent_state.values['last_agent']}")

if __name__ == '__main__':
    asyncio.run(main())
