import asyncio
import uuid

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

# Charger les variables d'environnement
load_dotenv()

from agent_supervisor.agent import agent


async def main():
    # Configuration initiale
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    run_config = RunnableConfig(callbacks=[], **config)

    while True:
        # Lire l'entrée utilisateur
        user_msg = input("User: ")
        print("\nAI: ", end="")

        async for event in agent.astream({"messages": [HumanMessage(content=user_msg)]}, config=run_config,
                                         stream_mode=["messages", "custom"]):
            event_type = event[0]
            payload = event[1]

            if event_type == "custom":
                print(payload, end="")

        # Afficher l'état de l'agent
        agent_state = agent.get_state(config)
        print(f"\n(Agent: {agent_state.values['last_agent']})\n")


if __name__ == '__main__':
    asyncio.run(main())
