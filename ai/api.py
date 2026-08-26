from fastapi import FastAPI
from ai.graph import stream_agent_response
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from fastapi.responses import StreamingResponse
from typing import Any


# AKA the payload
class InputMessage(BaseModel):
    thread_id: str
    messages: str


fastapi_app = FastAPI()



@fastapi_app.post("/generate_response")
def generate_response(message: InputMessage):

    streamed_response = stream_agent_response(
        input={'messages': [HumanMessage(content=message.messages)]},
        config={
            "configurable": {
                "thread_id": message.thread_id
            }
        }
    )

    return StreamingResponse(streamed_response, media_type='text/plain')