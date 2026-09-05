# Standard Libraies
from typing import Annotated, Sequence, TypedDict, Literal

# pydantic
from pydantic import BaseModel, Field

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, SystemMessage
from langsmith import traceable
from langgraph.config import get_stream_writer

class Classification(BaseModel):
    """A single routing decision: which agent to call with what query."""
    destination_node: Literal[
        "deep_research", 
        # "risk_assessment"
    ] = Field(
        description="The next specialized node to handle the task"
    )



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


# Supervisor Node
@traceable(run_type="chain")
def orchestrator_node(
        base_llm, 
        system_message: SystemMessage, 
        state: OrchestratorInputState
    ) -> OrchestratorOutputState:

    """Routes node based on user request"""

    writer = get_stream_writer()
    writer({"status": "Orchestrator: Routing user request..."})

    orchestrator_llm = base_llm.with_structured_output(Classification)

    conversation_history = [system_message] + state['messages']

    orchestrator_response: Classification = orchestrator_llm.invoke(input=conversation_history)


    return {
        "destination_node": orchestrator_response.destination_node
    }


# Routing function
def route_to_agent(state: OrchestratorOutputState) -> str:
    """For orchestrator routing to agents"""
    destination_node = state['destination_node']

    if destination_node == "deep_research":
        return "deep_research_path"
    
    # elif destination_node == "risk_assesssment":
    #     return "risk_assesssment_path"

