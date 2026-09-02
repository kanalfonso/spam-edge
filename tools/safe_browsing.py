from typing import TypedDict
from langgraph.config import get_stream_writer
from langgraph.graph import StateGraph, START, END
from langsmith import traceable
from langgraph.stream import ProtocolEvent, StreamChannel, StreamTransformer



# c) Safe browsing sub-agent
@traceable(run_type="chain")
def safe_browsing_test_node(state) -> dict:
    writer = get_stream_writer()
    writer({"status": "Conducting Safe Browsing API test..."})
    return {"messages": f"The LLM chose to use `{state['classification'].destination_node}`. Went to the safe browsing node"}




if __name__ == "__main__":

    class CustomTransformer(StreamTransformer):
        required_stream_modes = ("custom", "updates")

        def __init__(self, scope: tuple[str, ...] = ()) -> None:
            super().__init__(scope)
            self.log = StreamChannel()

        def init(self) -> dict:
            return {"custom": self.log}

        def process(self, event: ProtocolEvent) -> bool:
            if event["method"] == "custom":
                self.log.push(event["params"]["data"])
            return True


    class MockClassification:
        destination_node = "safe_browsing_test"


    class State(TypedDict):
        classification: MockClassification
        messages: str


    builder = StateGraph(State)
    builder.add_node("safe_browsing_test", safe_browsing_test_node)
    builder.add_edge(START, "safe_browsing_test")
    builder.add_edge("safe_browsing_test", END)

    app = builder.compile()


    mock_state = {
        "classification": MockClassification(),
        "messages": "test"
    }

    # Stream custom writer outputs and node state updates natively
    for mode, chunk in app.stream(
        mock_state, 
        stream_mode=["custom", "updates"]
    ):
        if mode == "custom":
            print(f"[STATUS]: {chunk['status']}")
            
        elif mode == "updates":
            for node_name, node_output in chunk.items():
                if node_name == "orchestrator_node":
                    continue
                
                msg = node_output.get("messages")
                if msg:
                    print(f"[MESSAGE]: {msg}")