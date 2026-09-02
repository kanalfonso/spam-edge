from langgraph.config import get_stream_writer
from langchain.tools import tool, ToolRuntime

def write_tool_status(runtime: ToolRuntime, message: str):
    if not runtime.state.get("enable_tool_writer", False):
        return

    writer = get_stream_writer()
    writer({"status": message})