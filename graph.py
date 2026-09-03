# Standard Libraies
from typing import Annotated, Sequence, TypedDict, Literal
from dotenv import load_dotenv

from pydantic import BaseModel, Field


# Langchain imports
from langchain_ollama import ChatOllama
from langchain_core.messages import (BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage)

# Langraph imports
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.config import get_stream_writer


from langgraph.prebuilt import ToolNode
from pydantic import BaseModel

from langgraph.types import interrupt, Command
from langgraph.stream import ProtocolEvent, StreamChannel, StreamTransformer
from langgraph.checkpoint.memory import InMemorySaver


# for tracing
from langsmith import traceable


# tool imports
from tools.deep_research import arxiv_search_tool, duckduckgo_search_tool


# Configs
MODEL = "qwen2.5:7b"
load_dotenv()


# messages
with open('messages/orchestrator_sys_msg.txt', 'r') as file:
    ORCHESTRATOR_SYS_MSG_FILE = file.read()

ORCHESTRATOR_SYS_MSG = SystemMessage(content=ORCHESTRATOR_SYS_MSG_FILE)


base_llm = ChatOllama(
    model=MODEL,
    temperature=0,
    keep_alive="30m"
)


class CustomTransformer(StreamTransformer):
    required_stream_modes = ("custom",)

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
        self.log = StreamChannel()

    def init(self) -> dict:
        return {"custom": self.log}

    def process(self, event: ProtocolEvent) -> bool:
        if event["method"] == "custom":
            self.log.push(event["params"]["data"])
        return True



class Classification(BaseModel):
    """A single routing decision: which agent to call with what query."""
    destination_node: Literal["search_engine", "web_risk_test"] = Field(
        description="The next specialized node to handle the task"
    )


# Build the graph state
class GraphState(TypedDict):
    user_query: str         # prompt of user that initiates the workflow
    messages: Annotated[Sequence[BaseMessage], add_messages]
    classification: Classification
    interruption_response: str
    enable_tool_writer: bool



class InputState(TypedDict):
    """Schema of keys exposed to input state"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_query: str
    enable_tool_writer: bool


class OrchestratorInputState(InputState):
    """Schema of keys exposed to orchestrator input state"""
    pass


class OrchestratorOutputState(TypedDict):
    """Schema of keys exposed to orchestrator state"""
    classification: Classification



class OutputState(TypedDict):
    """Schema of keys exposed to output state"""
    final_response: BaseMessage



class QueryGenerationInputState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_query: str


class Query(BaseModel):
    search_query: str = Field(
        description="A concise, search-engine-optimized query based on the user's request"
    )


class QueryList(BaseModel):
    query_list: list[Query]

class QueryGenerationOutputState(TypedDict):
    query_list: list[Query]


search_tools = [duckduckgo_search_tool, arxiv_search_tool]
search_tool_node = ToolNode(search_tools)


# Step 1: Supervisor Node
@traceable(run_type="chain")
def orchestrator_node(state: OrchestratorInputState) -> OrchestratorOutputState:
    """Routes node based on user request"""

    writer = get_stream_writer()
    writer({"status": "Orchestrator: Routing user request..."})

    orchestrator_llm = base_llm.with_structured_output(Classification)

    conversation_history = [ORCHESTRATOR_SYS_MSG] + state['messages']

    orchestrator_response: Classification = orchestrator_llm.invoke(input=conversation_history)

    # return {"classifications": orchestrator_response.classifications}

    return {
        "classification": orchestrator_response
    }



# Step 2: Sub-agent nodes
def generate_search_queries_node(state: QueryGenerationInputState) -> QueryGenerationOutputState:
    

    ### insert llm logic to generate queries
    query_generator_llm = base_llm.with_structured_output(QueryList)

    conversation_history = [QUERY_GENERATOR_SYS_MSG] + state['messages']

    generated_queries: QueryList = query_generator_llm.invoke(input=conversation_history)


    return {
        "query_list": generated_queries
    }





### TODO:



# a) Search Engine sub-agent
@traceable(run_type="chain")
def search_engine_node(state: GraphState) -> dict:
    search_engine_llm = base_llm.bind_tools(tools=search_tools, tool_choice="any")

    writer = get_stream_writer()

    # interruption_response = interrupt("Do you want to proceed with search?")
    
    messages = state["messages"]

    # If we already have tool results, let the LLM
    # synthesize the final answer WITHOUT forcing another tool call.
    if isinstance(messages[-1], ToolMessage):

        writer({"status": "Search Node: Summarizing search tool results..."})

        response = base_llm.invoke(
            [
                SystemMessage(
                    content="Use the search results to answer the user's request. "
                            "Keep the answer concise."
                )
            ] + list(messages)
        )

    else:
        writer({"status": "Search Node: Using search engine tool..."})


        search_engine_llm = base_llm.bind_tools(
            tools=search_tools,
            tool_choice="any"
        )

        response = search_engine_llm.invoke(
            [
                SystemMessage(
                    content="Use the available search tools to find information "
                            "needed to answer the user's request."
                )
            ] + list(messages)
        )


    return {
        "messages": [response]
    }



# b) Web Risk sub-agent
@traceable(run_type="chain")
def web_risk_test_node(state: GraphState) -> dict:
    writer = get_stream_writer()
    writer({"status": "Conducting Web Risk API test..."})
    return {"messages": [AIMessage(f"The LLM chose to use `{state['classification'].destination_node}`. Went to the web risk node")]}



# Step 3: Routing Logic for Sub-agents
def route_to_agent(state: GraphState) -> str:
    """For orchestrator routing to agents"""
    classification = state['classification']

    if classification.destination_node == "search_engine":
        return "search_engine_path"
    
    elif classification.destination_node == "web_risk_test":
        return "web_risk_test_path"



# For continuing or ending loop
def should_continue(state: GraphState) -> str:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "search_tool_node_path"

    # Otherwise, we stop (reply to the user)
    return "end_path"


workflow = (
    StateGraph(GraphState, input_schema=InputState, output_schema=OutputState)
        .add_node("orchestrator_node", orchestrator_node)
        .add_node("search_engine_node", search_engine_node)

        .add_node("search_tool_node", search_tool_node)
        
        .add_node("web_risk_test_node", web_risk_test_node)

        
        # edges

        .add_edge(START, "orchestrator_node")


        # choose sub-agent
        .add_conditional_edges(
            source="orchestrator_node",
            path=route_to_agent,
            path_map={
                "search_engine_path": "search_engine_node",
                "web_risk_test_path": "web_risk_test_node"
            }
        )


        # call tool or immediately end
        .add_conditional_edges(
            source="search_engine_node",
            path=should_continue,
            path_map={
                "search_tool_node_path": "search_tool_node",
                "end_path": END
            }
        )


        # pass tool output back to 
        .add_edge("search_tool_node", "search_engine_node")
        .add_edge("web_risk_test_node", END)

        .compile(checkpointer=InMemorySaver())
)



def main():
    """Conversation loop"""

    png_data = workflow.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(png_data)
        print("Graph saved as graph.png")

    config = {"configurable": {"thread_id": "1"}}


    user_input = input("First message: ")

    while user_input != "exit":

        stream_input = {
            "messages": [HumanMessage(content=user_input)], 
            "user_query": user_input,
            "enable_tool_writer": True
        }

        while stream_input is not None: 
            stream = workflow.stream_events(
                input=stream_input,
                config=config,
                version='v3',
                transformers=[CustomTransformer]
            )        

            for name, item in stream.interleave("messages", "custom"):
                if name == "messages":
                    message = item

                    if message.node == "orchestrator_node":
                        continue

                    for token in message.text:
                        print(token, end="", flush=True)

                elif name == "custom":
                    print(f"\n[STATUS]: {item['status']}")
            


            if stream.interrupted:
                interrupt_info = stream.interrupts[0].value
                user_response = input(f"[INTERRUPT] {interrupt_info}: ")

                stream_input = Command(resume=user_response)

            else:
                # no more interrupts
                stream_input = None
        
        # 3. Prompt for next input to avoid infinite loop
        user_input = input("\nNext message: ")



# Conversation Flow
if __name__ == "__main__":
    main()

