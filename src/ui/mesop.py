import asyncio
import uuid
from dataclasses import field, dataclass
from typing import AsyncGenerator, Literal

from dotenv import load_dotenv

load_dotenv()

import mesop as me
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from langchain_core.runnables import RunnableConfig

from agent_supervisor.agent import agent

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
_STYLE_TITLE = me.Style(padding=me.Padding(left=10))
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
_STYLE_CHAT_INPUT_BOX = me.Style(
  padding=me.Padding(top=30), display="flex", flex_direction="row"
)
_STYLE_CHAT_BUTTON = me.Style(margin=me.Margin(top=8, left=8))
_STYLE_CHAT_BUBBLE_NAME = me.Style(
  font_weight="bold",
  font_size="13px",
  padding=me.Padding(left=15, right=15, bottom=5),
)
_STYLE_CHAT_BUBBLE_PLAINTEXT = me.Style(margin=me.Margin.symmetric(vertical=15))



run_config : RunnableConfig

def on_load(e: me.LoadEvent):
  me.set_theme_mode("system")
  config = {"configurable": {"thread_id": str(uuid.uuid4())}}
  global run_config
  run_config = RunnableConfig(callbacks=[], **config)

@dataclass
class ChatMessage:
    type: str = ""
    message : str = ""


@me.stateclass
class State:
  input: str = ""
  messages: list[ChatMessage] = field(default_factory=list)
  in_progress: bool = False


@me.page(
  security_policy=me.SecurityPolicy(
    allowed_iframe_parents=["https://mesop-dev.github.io"]
  ),
  path="/",
  title="Mesop Demo Chat",
  on_load=on_load,
)
def page():
    with me.box(style=_STYLE_APP_CONTAINER ):
        chat()
        input_text()


def _make_style_chat_bubble_wrapper(role):
    align_items = "end" if role == _ROLE_USER else "start"
    return me.Style(
        display="flex",
        flex_direction="column",
        align_items=align_items,
    )

def _make_chat_bubble_style(role) -> me.Style:
  """Generates styles for chat bubble.

  Args:
    role: Chat bubble background color depends on the role
  """
  background = (
    _COLOR_CHAT_BUBBLE_YOU if role == _ROLE_USER else _COLOR_CHAT_BUBBLE_BOT
  )
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
    state = me.state(State)

    with me.box(style=_STYLE_CHAT_BOX):
        for msg in state.messages:

                with me.box(style=_make_style_chat_bubble_wrapper(msg.type)):
                    with me.box(style=_make_chat_bubble_style(msg.type)):
                        if msg.type == "ai":
                            me.text("Agent", style=_STYLE_CHAT_BUBBLE_PLAINTEXT)
                            me.markdown(msg.message)

                        if msg.type == "human":
                            me.text("Human", style=_STYLE_CHAT_BUBBLE_PLAINTEXT)
                            me.markdown(msg.message)
        if state.in_progress:
            with me.box(key="scroll-to", style=me.Style(height=300)):
                pass

async def on_input_enter(action: me.InputEnterEvent):
    state = me.state(State)
    state.input = action.value

    if state.messages is None:
        state.messages = []
    hm = ChatMessage()
    hm.type = "human"
    hm.message = state.input
    state.messages.append(hm)
    yield

    cm = ChatMessage()
    cm.type = "ai"
    cm.message = ""
    state.messages.append(cm)
    state.in_progress = True
    async for event in transform(state.input):
        cm.message += event
        me.scroll_into_view(key="scroll-to")
        yield
    state.in_progress = False

def input_text():
    with me.box():
        with me.box():
            me.input(
                label="Message",
                on_enter=on_input_enter,
                style=_STYLE_CHAT_INPUT
            )


async def transform(user_msg: str) -> AsyncGenerator[str, None]:
    async for event in agent.astream(
            {"messages": [HumanMessage(content=user_msg)]},
            config=run_config,
            stream_mode=["messages", "custom"]):

        event_type = event[0]
        payload = event[1]

        if event_type == "messages":
            yield payload[0].content

