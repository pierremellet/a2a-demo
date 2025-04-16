import asyncio
import uuid

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage, AIMessageChunk
from langchain_core.runnables import RunnableConfig

load_dotenv()

from agent_supervisor.agent import agent


async def main():
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    run_config = RunnableConfig(callbacks=[], **config)

    while True:
        user_msg = input("User : ")
        print("AI ", end="")
        agent_name_displayed = False
        async for msg, metadata in agent.astream(
                {"messages": [
                    HumanMessage(content=user_msg)
                ]},
                stream_mode="messages",
                config=run_config):
            if (
                    msg.content
                    and not isinstance(msg, HumanMessage)
                    and not isinstance(msg, ToolMessage)
            ):
                    if not agent_name_displayed:
                        if 'active_agent' in agent.get_state(config).values and agent.get_state(config).values['active_agent'] is not None:
                            print(f"({agent.get_state(config).values['active_agent']}) : ", end="")
                            agent_name_displayed = True
                    else:
                        if isinstance(msg, AIMessageChunk):
                            print(msg.content, end="")

        print(f"\nState : ")
        print(agent.get_state(config).values)
        print("\n")


if __name__ == '__main__':
    asyncio.run(main())
