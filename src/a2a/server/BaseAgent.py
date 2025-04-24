from abc import ABC, abstractmethod
from typing import AsyncIterable, Dict, Any

from langgraph.graph.graph import CompiledGraph

from a2a.common.common import ResponseFormat
from a2a.common.types import AgentSkill, AgentCard, AgentCapabilities


class BaseAgent(ABC):

    def __init__(self):
        self._agent = None

    @abstractmethod
    def get_agent_capabilites(self) -> AgentCapabilities :
        raise NotImplemented()

    @abstractmethod
    def get_agent_card(self) -> AgentCard:
        raise NotImplemented()

    @abstractmethod
    def get_agent_skills(self) -> AgentSkill:
        raise NotImplemented()

    def set_agent(self, agent: CompiledGraph):
        self._agent = agent

    async def async_invoke(self, query, sessionId) -> Dict[str, Any]:
        config = {"configurable": {"thread_id": sessionId}}
        await self._agent.ainvoke({"messages": [("user", query)]}, config)
        return self.get_agent_response(config)

    async def async_stream(self, query, sessionId) -> AsyncIterable[Dict[str, Any]]:
        inputs = {"messages": [("user", query)]}
        config = {"configurable": {"thread_id": sessionId}}

        buffer = ""
        in_message_value = False

        async for event in self._agent.astream(inputs, config, stream_mode=["values", "messages"]):
            type = event[0]
            payload = event[1]

            if type == "values" and "structured_response" in payload:
                structured_response = payload["structured_response"]
                if structured_response and isinstance(structured_response, ResponseFormat):
                    if structured_response.status == "input_required":
                        yield {
                            "is_task_complete": False,
                            "require_user_input": True,
                            "content": structured_response.message
                        }
                    elif structured_response.status == "error":
                        yield {
                            "is_task_complete": False,
                            "require_user_input": True,
                            "content": structured_response.message
                        }
                    elif structured_response.status == "completed":
                        yield {
                            "is_task_complete": True,
                            "require_user_input": False,
                            "content": structured_response.message
                        }

                yield {
                    "is_task_complete": False,
                    "require_user_input": True,
                    "content": "We are unable to process your request at the moment. Please try again.",
            }


            if type=="messages" :
                buffer += payload[0].content

                if '"message":"' in buffer and not in_message_value:
                    buffer = buffer.split('"message":"', 1)[1]
                    in_message_value = True

                if in_message_value:
                    # Divise le buffer à la fin de la valeur du message
                    if '"' in buffer:
                        message_value, remaining = buffer.split('"', 1)
                        yield {
                            "is_task_complete": False,
                            "require_user_input": False,
                            "content": message_value,
                        }
                        # Arrête si nous avons atteint la fin de la valeur du message
                        #break
                    else:
                        yield {
                            "is_task_complete": False,
                            "require_user_input": False,
                            "content": buffer,
                        }
                        buffer = ""

