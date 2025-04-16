from abc import ABC, abstractmethod
from typing import AsyncIterable, Dict, Any

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph.graph import CompiledGraph

from a2a.common.common import ResponseFormat
from a2a.common.types import AgentSkill, AgentCard, AgentCapabilities


class BaseAgent(ABC):

    def __init__(self):
        self.agent = None

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
        self.agent = agent

    def invoke(self, query, sessionId) -> str:
        config = {"configurable": {"thread_id": sessionId}}
        self.agent.invoke({"messages": [("user", query)]}, config)
        return self.get_agent_response(config)

    async def stream(self, query, sessionId) -> AsyncIterable[Dict[str, Any]]:
        inputs = {"messages": [("user", query)]}
        config = {"configurable": {"thread_id": sessionId}}

        for item in self.agent.stream(inputs, config, stream_mode="values"):
            message = item["messages"][-1]
            if (
                    isinstance(message, AIMessage)
                    and message.tool_calls
                    and len(message.tool_calls) > 0
            ):
                yield {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": "Looking up the exchange rates...",
                }
            elif isinstance(message, ToolMessage):
                yield {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": "Processing the exchange rates..",
                }

        yield self.get_agent_response(config)

    def get_agent_response(self, config):
        current_state = self.agent.get_state(config)
        structured_response = current_state.values.get('structured_response')
        if structured_response and isinstance(structured_response, ResponseFormat):
            if structured_response.status == "input_required":
                return {
                    "is_task_complete": False,
                    "require_user_input": True,
                    "content": structured_response.message
                }
            elif structured_response.status == "error":
                return {
                    "is_task_complete": False,
                    "require_user_input": True,
                    "content": structured_response.message
                }
            elif structured_response.status == "completed":
                return {
                    "is_task_complete": True,
                    "require_user_input": False,
                    "content": structured_response.message
                }

        return {
            "is_task_complete": False,
            "require_user_input": True,
            "content": "We are unable to process your request at the moment. Please try again.",
        }