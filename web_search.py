from langchain.tools import tool, ToolRuntime
from pydantic import Field, BaseModel
from langsmith import traceable
from tools.tool_utils import write_tool_status


class SearchArgsSchema(BaseModel):
    """This tool accepts only runtime as an argument"""
    # model_config = ConfigDict(arbitrary_types_allowed=True)

    # runtime: ToolRuntime

    search_query: str = Field(
        description="A concise, search-engine-optimized query based on the user's request"
    )



@tool(
        name_or_callable="arxiv_search_tool", 
        description="Use this tool to conduct a deep-dive or academic research on user request", 
        args_schema=SearchArgsSchema
    )
@traceable(run_type="tool")
def arxiv_search_tool(runtime: ToolRuntime, search_query: str) -> str:
    """Displays search results of the user request via arXiv"""

    write_tool_status(
        runtime,
        "Tool: Searching using arxiv"
    )
    
    return f"Used arXiv to search for {runtime.state['user_query']}. Processed by LLM: {search_query}. Latest scam is smishing."





@tool(
        name_or_callable="duckduckgo_search_tool", 
        description="Uses the DuckDuckGo search engine for basic user request", 
        args_schema=SearchArgsSchema
    )
@traceable(run_type="tool")
def duckduckgo_search_tool(runtime: ToolRuntime, search_query: str) -> str:
    """
    Displays search results of the user request via DuckDuckGo.

    No `args` passed to this function
    """


    write_tool_status(
        runtime,
        "Tool: Searching using DuckDuckGo"
    )


    return f"Used DuckDuckGo to search for {runtime.state['user_query']}. Processed by LLM: {search_query}. Latest scam is phishing"




if __name__ == "__main__":
    from unittest.mock import MagicMock
    
    mock_state = {"user_query": "What are recent online scams?"}

    # Create a mock runtime object with the required state dictionary
    runtime_instance = ToolRuntime(
        context={},           # Positional arg 1: context
        config={},            # Positional arg 2: config
        stream_writer=MagicMock(), # Positional arg 3: stream_writer
        tool_call_id="test_id",    # Positional arg 4: tool_call_id
        store=MagicMock(),         # Positional arg 5: store
        state=mock_state      # Keyword arg / field: state
    )

    result_arxiv = arxiv_search_tool.invoke({
        "runtime": runtime_instance,
        "search_query": "latest online scams"
    })

    print(result_arxiv)
    # Output: Used arXiv to search for What are recent online scams?. Processed by LLM: latest online scams. Latest scam is smishing.

    result_ddg = duckduckgo_search_tool.invoke({
        "runtime": runtime_instance,
        "search_query": "latest online scams"
    })

    print(result_ddg)