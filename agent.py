# Standard Libraies
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv

# Langchain imports
from langchain_ollama import ChatOllama
from langchain_core.messages import (BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage)


# Langraph imports
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END


####### CONFIGS #######
SYSTEM_MESSAGE_PATH = "messages/system_message.txt" 
MODEL = "qwen2.5:7b"

with open(SYSTEM_MESSAGE_PATH, "r") as file:
    system_message_raw = file.read()

    SYSTEM_MESSAGE = SystemMessage(content=system_message_raw)


####### CONFIGS #######

load_dotenv()

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

    return {'messages': response.content}



## Create graph ##
graph = StateGraph(AgentState)

graph.add_node("model_node", model_node)
graph.add_edge(START, "model_node")
graph.add_edge("model_node", END)

app = graph.compile()
## Create graph ##



## Check agent flowchart ##
png_data = app.get_graph().draw_mermaid_png()

with open("graph.png", "wb") as f:
    f.write(png_data)

## Check agent flowchart ##


def main():

    user_input = input("First message: ")

    while user_input != 'exit':
        msg = {'messages': [HumanMessage(content=user_input)]}
        app.invoke(msg)

        user_input = input("What's up: ")


if __name__ == '__main__':
    main()

