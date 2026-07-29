from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from langchain_core.runnables import Runnable
from langgraph.store.sqlite import SqliteStore  # type: ignore[import-not-found]
from langchain_ollama import ChatOllama
from typing_extensions import TypedDict


@dataclass
class Context:
    user_id: str


class UserInfo(TypedDict):
    name: str


llm = ChatOllama(
    model="qwen2.5:7b"
)


@tool
def save_user_info(user_info: UserInfo, runtime: ToolRuntime[Context]) -> str:
    """Save user info."""
    assert runtime.store is not None
    runtime.store.put(("users",), runtime.context.user_id, dict(user_info))
    return "Successfully saved user info."


DB_PATH = "chat_history.db"

with SqliteStore.from_conn_string(DB_PATH) as store:
    store.setup()
    agent: Runnable = create_agent(
        llm,
        tools=[save_user_info],
        store=store,
        context_schema=Context,
    )

    agent.invoke(
        {"messages": [{"role": "user", "content": "My name is John Smith"}]},
        context=Context(user_id="user_123"),
    )


print("done")