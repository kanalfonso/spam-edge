from ddgs import DDGS
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
import time
from langchain_core.messages import AIMessageChunk, ToolMessage

# results = DDGS().text("The Latest AI News", max_results=50)
# print(results)


# from langchain.tools import tool, ToolRuntime

llm = ChatOllama(
    model="qwen2.5:7b",
    temperature=0,
    keep_alive="30m"
)


# from google.oauth2.service_account import Credentials
# from dotenv import load_dotenv
# import os
# import json


# load_dotenv()


# GCP_SERVICE_ACCOUNT_CREDENTIALS = json.loads(os.getenv('GCP_SERVICE_ACCOUNT_CREDENTIALS'))

# # print(type(GCP_SERVICE_ACCOUNT_CREDENTIALS))
# CREDENTIALS = Credentials.from_service_account_info(
#     GCP_SERVICE_ACCOUNT_CREDENTIALS,
#     scopes=['https://www.googleapis.com/auth/cloud-platform']
# )


# from langchain_google_genai import ChatGoogleGenerativeAI

# gemini_model = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash-lite",
#     project="globe-eds-do-vina-dv",
#     location="us-central1",  # Optional, defaults to us-central1
#     temperature=1.0,  # Gemini 3.0+ defaults to 1.0
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
#     credentials=CREDENTIALS
#     # other params...
# )



@tool(return_direct=True)
def get_weather(city: str, runtime: ToolRuntime) -> str:
    """Get weather for a given city."""
    writer = runtime.stream_writer
    start = time.perf_counter()

    writer(f"\n[0.00s] Looking up data for city: {city}\n")
    time.sleep(1)

    elapsed = time.perf_counter() - start
    writer(f"[{elapsed:.2f}s] Acquired data for city: {city}\n")

    return f"It's always sunny in {city}!"


agent = create_agent(
    llm,
    tools=[get_weather],
)


# result = agent.stream(
#     {
#         "messages": [
#             {
#                 "role": "user",
#                 "content": "What is the weather today in Manila?"
#             }
#         ]
#     },
#     stream_mode=["custom", "updates", "messages"],
# )

# for mode, chunk in result:
#     print(f"[{mode}] {chunk}", flush=True)

result = agent.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "What is the weather today in Manila?"
            }
        ]
    },
    # Included 'messages' mode to capture model token deltas
    stream_mode=["custom", "updates", "messages"],
)

for mode, chunk in result:
    if mode == "messages":
        message_chunk, metadata = chunk
        
        # 1. Print AI tokens if the model generates text
        if isinstance(message_chunk, AIMessageChunk) and message_chunk.content:
            print(message_chunk.content, end="", flush=True)
            
        # 2. Print Tool response directly when return_direct=True
        elif isinstance(message_chunk, ToolMessage) and message_chunk.content:
            print(message_chunk.content, end="", flush=True)
            
    elif mode == "custom":
        # Stream events sent via runtime.stream_writer in tools
        print(chunk, end="", flush=True)

    elif mode == "updates":
        pass


print("\nend")