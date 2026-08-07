# refe

# Standard Libraies
import os
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
import time

# Langchain imports
from langchain_ollama import ChatOllama
from langchain_core.messages import (BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage)


# Langraph imports
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END


from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore


# for annotation
from langgraph.graph.state import CompiledStateGraph
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from langchain_core.messages import AIMessageChunk
import json
import asyncio

load_dotenv()

####### CONFIGS #######
SYSTEM_MESSAGE_PATH = "messages/system_message.txt" 
MODEL = "qwen2.5:7b"
DB_URI = "postgresql://postgres:postgres@localhost:5432/chathistory_db"
CONFIG = {
    "configurable": {
        "thread_id": "thread-1"
    }
}
####### CONFIGS #######


with open(SYSTEM_MESSAGE_PATH, "r") as file:
    system_message_raw = file.read()

    SYSTEM_MESSAGE = SystemMessage(content=system_message_raw)


llm = ChatOllama(
    model=MODEL,
    temperature=0,
    keep_alive="30m"
)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def model_node(state: AgentState) -> dict:

    # writer = get_stream_writer()
    # writer({'status': 'Model is generating a response...'})

    conversation_history = [SYSTEM_MESSAGE] + state['messages']

    # full_response = ""

    # for chunk in llm.stream(conversation_history):
    #     full_response += chunk.content

    # return {'messages': [AIMessage(content=full_response)]}

    response = llm.invoke(input=conversation_history)


    return {'messages': [AIMessage(response.content)]}


## Create graph ##
graph = (
    StateGraph(AgentState)
        .add_node("model_node", model_node)
        .add_edge(START, "model_node")
        .add_edge("model_node", END)
)


start = time.time()

def stream_agent_response(
        input: Sequence[BaseMessage], 
        config: dict[str, any]
    ):
    
    # use checkpointer to elegantly manage convo history
    with PostgresSaver.from_conn_string(DB_URI) as checkpointer:

        # checkpointer.setup()

        # compile graph
        compiled_graph = graph.compile(checkpointer=checkpointer)

        for chunk in compiled_graph.stream(
            input=input, 
            config=config,
            stream_mode=["messages"],
            version="v2"
        ):

            # streaming LLM outputs
            if chunk["type"] == "messages":
                message_chunk, metadata = chunk["data"]

                # Filter for streaming chunks only (ignores full AIMessage objects)
                if isinstance(message_chunk, AIMessageChunk) and message_chunk.content:
                    # print(message_chunk.content, end="", flush=True)
                    
                    yield message_chunk.content
                    # print(f"{time.time() - start:.2f}s: {message_chunk.content!r}")




if __name__ == '__main__':

    # Conversation flow
    user_input = input("First message: ")

    while user_input != 'exit':
        msg = {'messages': [HumanMessage(content=user_input)]}

        streamed_response = stream_agent_response(
            input=msg,
            config=CONFIG
        )

        print("Assistant: ", end="", flush=True)

        for chunk in streamed_response:
            print(chunk, end="", flush=True)

        print() 

        user_input = input("You: ")
        

