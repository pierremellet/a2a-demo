from dotenv import load_dotenv

from a2a.server.base_agent import BaseAgent
from a2a.server.server import A2AServer
from a2a.server.task_manager import logger
from a2a.utils.push_notification_auth import PushNotificationSenderAuth
from a2a.server.agent_task_manager import AgentTaskManager
from agent_coach.agent import CoachAgent

load_dotenv()


_HOST = "localhost"
_PORT = "8000"


notification_sender_auth = PushNotificationSenderAuth()
notification_sender_auth.generate_jwk()
agent : BaseAgent = CoachAgent()

server = A2AServer(
    agent_card=agent.get_agent_card(),
    task_manager=AgentTaskManager(agent=agent, notification_sender_auth=notification_sender_auth),
    host=_HOST,
    port=int(_PORT),
)

server.app.add_route(
    "/.well-known/jwks.json", notification_sender_auth.handle_jwks_endpoint, methods=["GET"]
)

logger.info(f"Starting server on {_HOST}:{_PORT}")
server.start()