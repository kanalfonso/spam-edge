# refe

# Standard Libraies
import os
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv

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
    temperature=0
)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def model_node(state: AgentState) -> dict:

    conversation_history = [SYSTEM_MESSAGE] + state['messages']

    response = llm.invoke(input=conversation_history)

    print(f"AI: {response.content}")

    return {'messages': AIMessage(response.content)}


## Create graph ##
graph = StateGraph(AgentState)

graph.add_node("model_node", model_node)
graph.add_edge(START, "model_node")
graph.add_edge("model_node", END)
## Create graph ##


def main(app: CompiledStateGraph, config: dict[str, any]):
    """Conversation flow"""
    user_input = input("First message: ")

    while user_input != 'exit':
        msg = {'messages': [HumanMessage(content=user_input)]}
        app.invoke(
            input=msg,
            config=config
        )

        user_input = input("What's up: ")



# use checkpointer to elegantly manage convo history
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:

    # checkpointer.setup()

    # compile graph
    app = graph.compile(checkpointer=checkpointer)


    ## Display workflow ##
    png_data = app.get_graph().draw_mermaid_png()


    with open("graph.png", "wb") as f:
        f.write(png_data)
    ## Display workflow ##



    if __name__ == '__main__':

        # pass app and config to the file
        main(app=app, config=CONFIG)

