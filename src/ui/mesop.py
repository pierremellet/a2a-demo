import json
import uuid
from dataclasses import field, dataclass
from typing import AsyncGenerator, Literal

from dotenv import load_dotenv

from agent_supervisor.utils import MessageEvent

load_dotenv()
import mesop as me
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from agent_supervisor.agent import supervisor_agent


Role = Literal["ai", "human"]

_ROLE_USER = "human"
_ROLE_ASSISTANT = "ai"
_BOT_USER_DEFAULT = "mesop-bot"

_COLOR_BACKGROUND = me.theme_var("background")
_COLOR_CHAT_BUBBLE_YOU = me.theme_var("surface-container-low")
_COLOR_CHAT_BUBBLE_BOT = me.theme_var("secondary-container")

_DEFAULT_PADDING = me.Padding.all(20)
_DEFAULT_BORDER_SIDE = me.BorderSide(
    width="1px", style="solid", color=me.theme_var("secondary-fixed")
)

_LABEL_BUTTON = "send"
_LABEL_BUTTON_IN_PROGRESS = "pending"
_LABEL_INPUT = "Enter your prompt"

_STYLE_APP_CONTAINER = me.Style(
    background=_COLOR_BACKGROUND,
    display="flex",
    flex_direction="column",
    height="100%",
    margin=me.Margin.symmetric(vertical=0, horizontal="auto"),
    width="min(1024px, 100%)",
    box_shadow=("0 3px 1px -2px #0003, 0 2px 2px #00000024, 0 1px 5px #0000001f"),
    padding=me.Padding(top=20, left=20, right=20),
)
_STYLE_TITLE = me.Style(padding=me.Padding(bottom=5), font_size="2rem")
_STYLE_CHAT_BOX = me.Style(
    flex_grow=1,
    overflow_y="scroll",
    padding=_DEFAULT_PADDING,
    margin=me.Margin(bottom=20),
    border_radius="10px",
    border=me.Border(
        left=_DEFAULT_BORDER_SIDE,
        right=_DEFAULT_BORDER_SIDE,
        top=_DEFAULT_BORDER_SIDE,
        bottom=_DEFAULT_BORDER_SIDE,
    ),
)
_STYLE_CHAT_INPUT = me.Style(width="100%")
_STYLE_CHAT_INPUT_BOX = me.Style(padding=me.Padding(top=30), display="flex", flex_direction="row")
_STYLE_CHAT_BUTTON = me.Style(margin=me.Margin(top=8, left=8))
_STYLE_CHAT_BUBBLE_NAME = me.Style(font_weight="bold", font_size="13px", padding=me.Padding(left=15, right=15, bottom=5))
_STYLE_CHAT_BUBBLE_PLAINTEXT = me.Style(margin=me.Margin.symmetric(vertical=15), font_weight="bold")

run_config: RunnableConfig

def on_load(e: me.LoadEvent):
    me.set_theme_mode("light")
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    global run_config
    run_config = RunnableConfig(callbacks=[], **config)

@dataclass
class ChatMessage:
    id: str = ""
    type: str = ""
    message: str = ""
    active_agent: str = ""
    task_id: str = ""

@me.stateclass
class State:
    input: str = ""
    messages: list[ChatMessage] = field(default_factory=list)
    in_progress: bool = False
    agent_state: dict = field(default_factory=dict)

@me.page(
    security_policy=me.SecurityPolicy(),
    path="/",
    title="A2A Demo Chat",
    on_load=on_load
)
def page():
    with me.box(style=_STYLE_APP_CONTAINER):
        me.text("Assistant", style=_STYLE_TITLE)
        chat()
        input_text()
        state()

def state():
    ui_state = me.state(State)
    with me.expansion_panel(
            key="state",
            title="Agent state",
            description="...",
            icon="database",
            disabled=False,
            hide_toggle=False,
    ):
        str_state = format_agent_state(ui_state.agent_state)
        me.markdown(str_state)

def format_agent_state(agent_state: dict) -> str:
    str_state = ""
    for item, value in agent_state.items():
        if isinstance(value, str) or value is None:
            str_state += f"- **{item}** : {value}\n"
        elif isinstance(value, list):
            str_state += f"- **{item}** :\n\t- size : {len(value)}\n\t- last message : {value[-1].content}\n"
    return str_state

def _make_style_chat_bubble_wrapper(role: Role) -> me.Style:
    align_items = "end" if role == _ROLE_USER else "start"
    return me.Style(display="flex", flex_direction="column", align_items=align_items)

def _make_chat_bubble_style(role: Role) -> me.Style:
    background = _COLOR_CHAT_BUBBLE_YOU if role == _ROLE_USER else _COLOR_CHAT_BUBBLE_BOT
    return me.Style(
        width="80%",
        font_size="16px",
        line_height="1.5",
        background=background,
        border_radius="15px",
        padding=me.Padding(right=15, left=15, bottom=3),
        margin=me.Margin(bottom=10),
        border=me.Border(
            left=_DEFAULT_BORDER_SIDE,
            right=_DEFAULT_BORDER_SIDE,
            top=_DEFAULT_BORDER_SIDE,
            bottom=_DEFAULT_BORDER_SIDE,
        ),
    )

def chat():
    ui_state = me.state(State)
    with me.box(style=_STYLE_CHAT_BOX):
        for msg in ui_state.messages:
            with me.box(style=_make_style_chat_bubble_wrapper(msg.type)):
                with me.box(style=_make_chat_bubble_style(msg.type)):
                    role_text = "Agent" if msg.type == "ai" else "Human"
                    me.text(f"{role_text} : {msg.active_agent if msg.type == 'ai' else ''}", style=_STYLE_CHAT_BUBBLE_PLAINTEXT)
                    if msg.id : me.text(f"id : {msg.id}")
                    me.markdown(msg.message.replace("\\n", "<br/>"))

async def on_input_enter(action: me.InputEnterEvent):
    ui_state = me.state(State)
    ui_state.input = action.value

    if ui_state.messages is None:
        ui_state.messages = []

    ui_state.messages.append(ChatMessage(type="human", message=ui_state.input))
    yield

    ai_message = None

    async for event in transform(ui_state.input):

        if ai_message is None :
            ai_message = ChatMessage(type="ai", message="")
            ui_state.messages.append(ai_message)
            yield
            ai_message.message += event['content']
            ai_message.active_agent = event['agent']

        elif ai_message.active_agent == event['agent']:
            ai_message.message += event['content']

        elif ai_message.active_agent != event['agent']:
            ai_message = ChatMessage(type="ai", message="")
            ui_state.messages.append(ai_message)
            yield
            ai_message.message += event['content']
            ai_message.active_agent = event['agent']

        yield

    ui_state.agent_state = {key: value for key, value in supervisor_agent.get_state(config=run_config).values.items() if isinstance(value, (str, type(None)))}
    yield

    ui_state.in_progress = False
    yield

def input_text():
    with me.box():
        me.input(label="Message", on_enter=on_input_enter, style=_STYLE_CHAT_INPUT)

async def transform(user_msg: str) -> AsyncGenerator[MessageEvent, None]:
    async for event in supervisor_agent.astream({"messages": [HumanMessage(content=user_msg)]}, config=run_config, stream_mode=["custom"]):
        event_type, payload = event

        if event_type == "custom":
            yield payload
