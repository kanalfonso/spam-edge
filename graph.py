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

from langgraph.types import interrupt, Command, Send
from langgraph.stream import ProtocolEvent, StreamChannel, StreamTransformer
from langgraph.checkpoint.memory import InMemorySaver


# for tracing
from langsmith import traceable


# tool imports
from tools.deep_research import arxiv_search_tool, duckduckgo_search_tool



import operator


# Configs
MODEL = "qwen2.5:7b"
load_dotenv()


# Prompts
with open('prompts/01_orchestrator_node/01_system_message.txt', 'r') as file:
    ORCHESTRATOR_SYS_MSG_FILE = file.read()

with open('prompts/02_deep_research_node/01_query_generator_sys_msg.txt', 'r') as file:
    QUERY_GENERATOR_SYS_MSG_FILE = file.read()

with open('prompts/02_deep_research_node/02_deep_research_sys_msg.txt', 'r') as file:
    DEEP_RESEARCH_SYS_MSG_FILE = file.read()

with open('prompts/02_deep_research_node/03_summarizer_sys_msg.txt', 'r') as file:
    SUMMARIZER_SYS_MSG_FILE = file.read()



ORCHESTRATOR_SYS_MSG = SystemMessage(content=ORCHESTRATOR_SYS_MSG_FILE)
QUERY_GENERATOR_SYS_MSG = SystemMessage(content=QUERY_GENERATOR_SYS_MSG_FILE)
DEEP_RESEARCH_SYS_MSG = SystemMessage(content=DEEP_RESEARCH_SYS_MSG_FILE)
SUMMARIZER_SYS_MSG = SystemMessage(content=SUMMARIZER_SYS_MSG_FILE)




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


### Pydantic validation schemas
class Query(BaseModel):
    """Validation for search query"""
    search_query: str = Field(
        description="A concise, search-engine-optimized query based on the user's request"
    )


class QueryList(BaseModel):
    """Collection of search queries"""
    query_list: list[Query] = Field(..., min_length=1, max_length=3)


class Classification(BaseModel):
    """A single routing decision: which agent to call with what query."""
    destination_node: Literal[
        "deep_research", 
        # "risk_assessment"
    ] = Field(
        description="The next specialized node to handle the task"
    )



class ResearchFormat(BaseModel):
    research_results: str = Field(
        # min_length=100,  # Minimum character count
        # max_length=2000, # Maximum character count
        description="Report findings from the research topic."
    )


### States
class OverallState(TypedDict):
    
    # keys at inputstate
    enable_tool_writer: bool
    user_query: str         # prompt of user that initiates the workflow
    messages: Annotated[Sequence[BaseMessage], add_messages]
    # interruption_response: str

    # from orchestrator node
    classification: Literal[
        "deep_research", 
        # "risk_assesssment"
    ]

    # from QueryGeneratorOutputState
    query_list: list[str]


    # keys at output state
    final_response: BaseMessage



class InputState(TypedDict):
    """Input state accepted by the graph at invocation."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_query: str
    enable_tool_writer: bool


class OrchestratorInputState(TypedDict):
    """Input state required by the orchestrator node to make routing decisions."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_query: str



class OrchestratorOutputState(TypedDict):
    """Output state produced by the orchestrator node containing the routing classification."""
    destination_node: Literal[
        "deep_research", 
        # "risk_assesssment"
    ]



class OutputState(TypedDict):
    """Output state produced by the graph at invocation."""
    final_response: BaseMessage



class QueryGeneratorInputState(TypedDict):
    """Input state required by the `search_query_generator_node` function"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_query: str


class QueryGeneratorOutputState(TypedDict):
    """Output state produced by the `search_query_generator_node` function"""
    query_list: list[str]



class DeepResearchInputState(TypedDict):
    """Input state required by the `deep_research_node`"""
    query: str


class DeepResearchOutputState(TypedDict):
    """Output state produced by the `deep_research_node"""
    reseach_results: Annotated[list[str], operator.add]



class ResearchSummarizerOutputState(TypedDict):
    """Output state produced by the `research_summarizer_node"""

    summarized_results: str


### deep research tools
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


    return {
        "destination_node": orchestrator_response.destination_node
    }



def route_to_agent(state: OrchestratorOutputState) -> str:
    """For orchestrator routing to agents"""
    destination_node = state['destination_node']

    if destination_node == "deep_research":
        return "deep_research_path"
    
    # elif destination_node == "risk_assesssment":
    #     return "risk_assesssment_path"



# Sub-agents


def search_query_generator_node(state: QueryGeneratorInputState) -> QueryGeneratorOutputState:
    """Use LLM to generate search queries (max 3) based on user request"""

    ### insert llm logic to generate queries
    query_generator_llm = base_llm.with_structured_output(QueryList)

    # TODO: rethink if we should pass the entire convo `state['messages]` or just the user query `state['user_query]`
    conversation_history = [QUERY_GENERATOR_SYS_MSG] + state['messages']

    generated_queries: QueryList = query_generator_llm.invoke(input=conversation_history)


    return {
        "query_list": [query.search_query for query in generated_queries.query_list]
    }




def continue_to_deep_research(state: QueryGeneratorOutputState):
    """Fan out generated search queries to parallel deep research nodes."""

    return [
        Send("deep_research_node", {"query": query}) 
        for query in state['query_list']
    ]


def deep_research_node(state: DeepResearchInputState) -> DeepResearchOutputState:
    """Conduct deep research using LLM"""

    research_llm = base_llm.with_structured_output(ResearchFormat)

    # format the query
    query = f"Please make a report on the following research Topic: {state['query']}"
    
    conversation_history = [DEEP_RESEARCH_SYS_MSG] + [HumanMessage(query)]

    result: ResearchFormat = research_llm.invoke(input=conversation_history)

    return {
        "reseach_results": [result.research_results]
    }


def research_summarizer_node(state: DeepResearchOutputState) -> ResearchSummarizerOutputState:
    """Aggregate and summarize individual research results collected from parallel deep research nodes."""

    # 1. Access the aggregated list of results from state
    results = state.get("reseach_results", [])

    if not results:
        return {"summarized_results": "No research results available to summarize."}

    # 2. Format the combined research payload for the LLM
    formatted_results = "\n\n---\n\n".join(
        f"Result {i+1}:\n{res}" for i, res in enumerate(results)
    )

    prompt = f"Synthesize the following research findings:\n\n{formatted_results}"
    conversation_history = [SUMMARIZER_SYS_MSG, HumanMessage(content=prompt)]

    # 3. Call the base LLM (or a structured output LLM)
    response = base_llm.invoke(conversation_history)

    # 4. Return the finalized summary
    return {
        "summarized_results": response.content
    }




def main():

    workflow = (
        StateGraph(OverallState, input_schema=InputState, output_schema=OutputState)
            .add_node("orchestrator_node", orchestrator_node)
            .add_node("search_query_generator_node", search_query_generator_node)
            .add_node("deep_research_node", deep_research_node)
            .add_node("research_summarizer_node", research_summarizer_node)

            .add_edge(START, "orchestrator_node")
            
            .add_conditional_edges(
                source="orchestrator_node",
                path=route_to_agent,
                path_map={
                    "deep_research_path": "search_query_generator_node",
                    # "risk_assessment_path": risk_assessment_node
                }

            )
            

            # .add_edge("search_query_generator_node", "deep_research_node")
            .add_conditional_edges("search_query_generator_node", continue_to_deep_research, ["deep_research_node"])
            .add_edge("deep_research_node", "research_summarizer_node")
            .add_edge("research_summarizer_node", END)
            
            .compile(checkpointer=InMemorySaver())


    )


    png_data = workflow.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(png_data)
        print("Graph saved as graph.png")


    config = {"configurable": {"thread_id": "1"}}


    user_input = input("First message: ")

    while user_input != "exit":

        result = workflow.invoke(
            input={
                'messages': user_input,
                'user_query': user_input,
                'enable_tool_writer': True
            },
            config=config
        )

        print(f"AI Response: {result}")
        # 3. Prompt for next input to avoid infinite loop
        user_input = input("\nNext message: ")


if __name__ == "__main__":
    main()















# ### TODO:



# # a) Search Engine sub-agent
# @traceable(run_type="chain")
# def search_engine_node(state: GraphState) -> dict:
#     search_engine_llm = base_llm.bind_tools(tools=search_tools, tool_choice="any")

#     writer = get_stream_writer()

#     # interruption_response = interrupt("Do you want to proceed with search?")
    
#     messages = state["messages"]

#     # If we already have tool results, let the LLM
#     # synthesize the final answer WITHOUT forcing another tool call.
#     if isinstance(messages[-1], ToolMessage):

#         writer({"status": "Search Node: Summarizing search tool results..."})

#         response = base_llm.invoke(
#             [
#                 SystemMessage(
#                     content="Use the search results to answer the user's request. "
#                             "Keep the answer concise."
#                 )
#             ] + list(messages)
#         )

#     else:
#         writer({"status": "Search Node: Using search engine tool..."})


#         search_engine_llm = base_llm.bind_tools(
#             tools=search_tools,
#             tool_choice="any"
#         )

#         response = search_engine_llm.invoke(
#             [
#                 SystemMessage(
#                     content="Use the available search tools to find information "
#                             "needed to answer the user's request."
#                 )
#             ] + list(messages)
#         )


#     return {
#         "messages": [response]
#     }



# # b) Web Risk sub-agent
# @traceable(run_type="chain")
# def web_risk_test_node(state: GraphState) -> dict:
#     writer = get_stream_writer()
#     writer({"status": "Conducting Web Risk API test..."})
#     return {"messages": [AIMessage(f"The LLM chose to use `{state['classification'].destination_node}`. Went to the web risk node")]}



# # Step 3: Routing Logic for Sub-agents
# def route_to_agent(state: GraphState) -> str:
#     """For orchestrator routing to agents"""
#     classification = state['classification']

#     if classification.destination_node == "search_engine":
#         return "search_engine_path"
    
#     elif classification.destination_node == "web_risk_test":
#         return "web_risk_test_path"



# # For continuing or ending loop
# def should_continue(state: GraphState) -> str:
#     """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

#     messages = state["messages"]
#     last_message = messages[-1]

#     # If the LLM makes a tool call, then perform an action
#     if last_message.tool_calls:
#         return "search_tool_node_path"

#     # Otherwise, we stop (reply to the user)
#     return "end_path"


# workflow = (
#     StateGraph(GraphState, input_schema=InputState, output_schema=OutputState)
#         .add_node("orchestrator_node", orchestrator_node)
#         .add_node("search_engine_node", search_engine_node)

#         .add_node("search_tool_node", search_tool_node)
        
#         .add_node("web_risk_test_node", web_risk_test_node)

        
#         # edges

#         .add_edge(START, "orchestrator_node")


#         # choose sub-agent
#         .add_conditional_edges(
#             source="orchestrator_node",
#             path=route_to_agent,
#             path_map={
#                 "search_engine_path": "search_engine_node",
#                 "web_risk_test_path": "web_risk_test_node"
#             }
#         )


#         # call tool or immediately end
#         .add_conditional_edges(
#             source="search_engine_node",
#             path=should_continue,
#             path_map={
#                 "search_tool_node_path": "search_tool_node",
#                 "end_path": END
#             }
#         )


#         # pass tool output back to 
#         .add_edge("search_tool_node", "search_engine_node")
#         .add_edge("web_risk_test_node", END)

#         .compile(checkpointer=InMemorySaver())
# )



# def main():
#     """Conversation loop"""

#     png_data = workflow.get_graph().draw_mermaid_png()
#     with open("graph.png", "wb") as f:
#         f.write(png_data)
#         print("Graph saved as graph.png")

#     config = {"configurable": {"thread_id": "1"}}


#     user_input = input("First message: ")

#     while user_input != "exit":

#         stream_input = {
#             "messages": [HumanMessage(content=user_input)], 
#             "user_query": user_input,
#             "enable_tool_writer": True
#         }

#         while stream_input is not None: 
#             stream = workflow.stream_events(
#                 input=stream_input,
#                 config=config,
#                 version='v3',
#                 transformers=[CustomTransformer]
#             )        

#             for name, item in stream.interleave("messages", "custom"):
#                 if name == "messages":
#                     message = item

#                     if message.node == "orchestrator_node":
#                         continue

#                     for token in message.text:
#                         print(token, end="", flush=True)

#                 elif name == "custom":
#                     print(f"\n[STATUS]: {item['status']}")
            


#             if stream.interrupted:
#                 interrupt_info = stream.interrupts[0].value
#                 user_response = input(f"[INTERRUPT] {interrupt_info}: ")

#                 stream_input = Command(resume=user_response)

#             else:
#                 # no more interrupts
#                 stream_input = None
        
#         # 3. Prompt for next input to avoid infinite loop
#         user_input = input("\nNext message: ")



# # Conversation Flow
# if __name__ == "__main__":
#     main()

