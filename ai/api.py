from fastapi import FastAPI
from agent import graph
from pydantic import BaseModel
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import (BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage)
from langgraph.checkpoint.postgres import PostgresSaver
from typing import Any
import requests

class InputMessage(BaseModel):
    thread_id: str
    messages: str

fastapi_app = FastAPI()



@fastapi_app.post("/generate_response", response_model=dict[str, Any])
def generate_response(message: InputMessage):

    DB_URI = "postgresql://postgres:postgres@localhost:5432/chathistory_db"
    CONFIG = {
        "configurable": {
            "thread_id": "thread-1"
        }
    }

    with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
        app = graph.compile(checkpointer=checkpointer)

        response = app.invoke(
            input=message,
            config=CONFIG
        )

    latest_message = response["messages"][-1]
    return latest_message.model_dump()
