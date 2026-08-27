# Standard Libraies
from typing import Annotated, Sequence, TypedDict, Literal
from dotenv import load_dotenv

from pydantic import BaseModel, Field


# Langchain imports
from langchain_ollama import ChatOllama
from langchain_core.messages import (BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage, AIMessageChunk)
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent

# Langraph imports
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.config import get_stream_writer


# Configs
MODEL = "qwen2.5:7b"
load_dotenv()

SEARCH_PLANNER_SYSTEM_MESSAGE = SystemMessage(
    content=
    """
    You are a Search Planner responsible for converting a user's request into an effective search strategy.
    """
)

base_llm = ChatOllama(
    model=MODEL,
    temperature=0,
    keep_alive="30m"
)



class Classification(BaseModel):
    """A single routing decision: which agent to call with what query."""
    destination_node: Literal["search_engine", "safe_browsing_test", "web_risk_test"] = Field(
        description="The next specialized node to handle the task"
    )
    # query: str = Field(
    #     description="Query asked by the user"
    # )


# # Define structured output schema for the classifier
# class ClassificationResult(BaseModel):
#     """Result of classifying a user query into agent-specific sub-questions."""
#     classifications: list[Classification] = Field(
#         description="List of agents to invoke with their targeted sub-questions"
#     )



# Build the graph state
class GraphState(TypedDict):
    user_query: str         # prompt of user that initiates the workflow
    messages: Annotated[Sequence[BaseMessage], add_messages]
    # classifications: list[Classification]
    classification: Classification


# # Search Engine Tools
# @tool(name="duckduckgo_search_tool", description="Uses the DuckDuckGo search engine for basic user request")
# def duckduckgo_search_tool(runtime: ToolRuntime) -> str:
#     """Displays search results of the user request via DuckDuckGo"""

#     return f"Used DuckDuckGo to search for {runtime.state['user_query']}"


# @tool(name="arxiv_search_tool", description="Use this tool to conduct a deep-dive or academic research on user request")
# def arxiv_search_tool(runtime: ToolRuntime) -> str:
#     """Displays search results of the user request via arXiv"""

#     return f"Used arXiv to search for {runtime.state['user_query']}"


# Step 1: Supervisor Node
def orcherstrator_node(state: GraphState) -> dict:
    """Routes node based on user request"""

    writer = get_stream_writer()
    writer({"status": "Attempting to route user request..."})

    orchestrator_llm = base_llm.with_structured_output(Classification)

    conversation_history = [SEARCH_PLANNER_SYSTEM_MESSAGE] + state['messages']

    orchestrator_response = orchestrator_llm.invoke(input=conversation_history)

    # return {"classifications": orchestrator_response.classifications}

    return {"classification": orchestrator_response}



# Step 2: Sub-agent nodes

# a) Search Engine sub-agent
def search_engine_node(state: GraphState) -> dict:
    # search_engine_llm = base_llm.bind_tools(tools=[duckduckgo_search_tool, arxiv_search_tool])

    writer = get_stream_writer()
    writer({"status": "Using search engine..."})
    
    user_query = [HumanMessage(state['user_query'])]
    # search_engine_response = search_engine_llm.invoke(user_query)

    search_engine_response = base_llm.invoke(
        [SystemMessage("Keep answers short, just 2-3 sentences at most.")] + user_query
    )

    writer = get_stream_writer()
    writer({"status": "Processing with search engine..."})

    return {'messages': [AIMessage(search_engine_response.content)]}



# b) Web Risk sub-agent
def web_risk_test_node(state: GraphState) -> dict:
    writer = get_stream_writer()
    writer({"status": "Conducting Web Risk API test..."})
    return {"messages": [AIMessage(f"The LLM chose to use `{state['classification'].destination_node}`. Went to the web risk node")]}


# c) Safe browsing sub-agent 
def safe_browsing_test_node(state: GraphState) -> dict:
    writer = get_stream_writer()
    writer({"status": "Conducting Safe Browsing API test..."})
    return {"messages": [AIMessage(f"The LLM chose to use `{state['classification'].destination_node}`. Went to the safe browsing node")]}



# Step 3: Routing Logic
def decide_next_node(state: GraphState):
    classification = state['classification']

    if classification.destination_node == "search_engine":
        return "search_engine_path"
    
    elif classification.destination_node == "safe_browsing_test":
        return "safe_browsing_test_path"
    
    elif classification.destination_node == "web_risk_test":
        return "web_browsing_test_path"



workflow = (
    StateGraph(GraphState)
        .add_node("orchestrator_node", orcherstrator_node)
        .add_node("search_engine_node", search_engine_node)
        .add_node("safe_browsing_test_node", safe_browsing_test_node)
        .add_node("web_risk_test_node", web_risk_test_node)
        

        .add_edge(START, "orchestrator_node")
        .add_conditional_edges(
            source="orchestrator_node",
            path=decide_next_node,
            path_map={
                "search_engine_path": "search_engine_node",
                "safe_browsing_test_path": "safe_browsing_test_node",
                "web_browsing_test_path": "web_risk_test_node"
            }
        )

        .add_edge("search_engine_node", END)
        .add_edge("safe_browsing_test_node", END)
        .add_edge("web_risk_test_node", END)

        .compile()        
)





# # Conversation Flow
# if __name__ == "__main__":

#     # Conversation flow
#     user_input = input("First message: ")

#     while user_input != 'exit':
#         processed_message = [HumanMessage(content=user_input)]

#         for chunk in workflow.stream(
#             {"messages": processed_message},
#             stream_mode=["custom", "messages"],
#             version="v2",
#         ):
#             if chunk["type"] == "messages":
#                 message_chunk, metadata = chunk["data"]

#                 # Filter for streaming chunks only (ignores full AIMessage objects)
#                 if isinstance(message_chunk, AIMessageChunk) and message_chunk.content:
#                     print(message_chunk.content, end="", flush=True)

#             elif chunk["type"] == "custom":
#                 print(f"Status: {chunk['data']['status']}")


# Conversation Flow
if __name__ == "__main__":

    user_input = input("First message: ")

    while user_input != 'exit':
        processed_message = [HumanMessage(content=user_input)]

        for chunk in workflow.stream(
            {"messages": processed_message, "user_query": user_input},
            stream_mode=["custom", "messages"],
            version="v2",
        ):
            if chunk["type"] == "messages":
                message_chunk, metadata = chunk["data"]

                # 1. Ignore messages from the orchestrator node (prevents printing JSON)
                if metadata.get("langgraph_node") == "orchestrator_node":
                    continue

                # 2. Print messages from actual subagent nodes (handles AIMessage and AIMessageChunk)
                if isinstance(message_chunk, BaseMessage) and message_chunk.content:
                    print(message_chunk.content, end="", flush=True)

            elif chunk["type"] == "custom":
                # Add newlines around custom status updates for cleaner formatting
                print(f"\nStatus: {chunk['data']['status']}", flush=True)

        print("\n")  # Newline after workflow finishes
        
        # 3. Prompt for next input to avoid infinite loop
        user_input = input("Next message: ")