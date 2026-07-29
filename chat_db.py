from dataclasses import dataclass
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore  
from langgraph.runtime import Runtime  
import uuid
import asyncio
from langchain_ollama import ChatOllama
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

model = ChatOllama(
    model="qwen2.5:7b"
)

@dataclass
class Context:
    user_id: str

async def call_model(
    state: MessagesState,
    runtime: Runtime[Context],
):
    user_id = runtime.context.user_id  
    namespace = ("memories", user_id)
    memories = await runtime.store.asearch(namespace, query=str(state["messages"][-1].content))
    info = "\n".join([d.value["data"] for d in memories])
    system_msg = f"You are a helpful assistant talking to the user. User info: {info}"

    # Store new memories if the user asks the model to remember
    last_message = state["messages"][-1]
    if "remember" in last_message.content.lower():
        memory = "User name is Bob"
        await runtime.store.aput(namespace, str(uuid.uuid4()), {"data": memory})

    response = await model.ainvoke(
        [{"role": "system", "content": system_msg}] + state["messages"]
    )
    return {"messages": response}

DB_URI = "postgresql://postgres:postgres@localhost:5432/postgres?sslmode=disable"

async def main():

    async with (
        AsyncPostgresStore.from_conn_string(DB_URI) as store,
        AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer,
    ):
        # await store.setup()
        # await checkpointer.setup()

        builder = StateGraph(MessagesState, context_schema=Context)
        builder.add_node(call_model)
        builder.add_edge(START, "call_model")

        graph = builder.compile(
            checkpointer=checkpointer,
            store=store,
        )

        config = {"configurable": {"thread_id": "1"}}
        stream = await graph.astream_events(
            {"messages": [{"role": "user", "content": "Hi! Remember: my name is Bob"}]},
            config,
            version="v3",
            context=Context(user_id="1"),
        )
        async for message in stream.messages:
            async for token in message.text:
                print(token, end="", flush=True)

        config = {"configurable": {"thread_id": "2"}}
        stream = await graph.astream_events(
            {"messages": [{"role": "user", "content": "what is my name?"}]},
            config,
            version="v3",
            context=Context(user_id="1"),
        )
        async for message in stream.messages:
            async for token in message.text:
                print(token, end="", flush=True)


if __name__ == "__main__":
    asyncio.run(main())