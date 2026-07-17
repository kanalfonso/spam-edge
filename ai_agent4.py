from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv  
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import os 


load_dotenv()


# Step 0: Define model
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')


model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=GEMINI_API_KEY,
)

document_content = ""

# Step 1: Define Tools
@tool
def update_doc(content: str) -> str:
    """Updates content of document based on the user request"""
    global document_content

    document_content = content

    return f"Document content has been updated. Current content is:\n\n {document_content}"


@tool
def save_doc(filename: str) -> str:
    """Saves document to a .txt file"""

    global document_content

    if not filename.endswith('.txt'):
        filename = f"{filename}.txt"

    try:
        with open(filename, 'w') as file:
            file.write(document_content)

        print(f"\n💾 Document has been saved to: {filename}")
        return f"Document has been saved successfully to '{filename}'."   

    except Exception as e:
        return f"Error saving document: {str(e)}"


tools = [update_doc, save_doc]
tools_by_name = {tool.name: tool for tool in tools}

model_with_tools = model.bind_tools(tools=tools)
    

# Step 2: Define state
class AgentState:
    messages: Annotated[Sequence[BaseMessage], add_messages]


# Step 3: Define nodes

def model_node(state: AgentState) -> dict:
    system_prompt = SystemMessage(content=f"""
    You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.
    
    - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
    - If the user wants to save and finish, you need to use the 'save' tool.
    - Make sure to always show the current document state after modifications.
    
    The current document content is:{document_content}
    """)

    print("Entering model node...\n")
    # print(f"Message state: {state['messages']}")

    # # first run
    # if not state['messages']:

    #     user_input = input("What's your request: ")
    #     human_msg = HumanMessage(user_input)

    #     complete_msg = [system_prompt] + [human_msg]

    #     # AIMessage object
    #     response = model_with_tools.invoke(input=complete_msg)
        

    # # succeeding runs
    # else:

    if isinstance(state['messages'][-1], HumanMessage):

        complete_msg = [system_prompt] + [state['messages'][-1]]
        response = model_with_tools.invoke(input=complete_msg)
    
        ai_content = response.content

        if isinstance(ai_content, list):
            ai_content = ai_content[0]["text"]
        
        print(f"AI: {ai_content}")

        return {
            # ai response
            'messages': [response]
        }


    else:

        ai_content = state["messages"][-1].content

        if isinstance(ai_content, list):
            ai_content = ai_content[0]["text"]
        
        print(f"AI: {ai_content}")

        user_input = input("Document Feedback: ")

        human_msg = HumanMessage(user_input)

        complete_msg = [system_prompt] + state['messages'] + [human_msg]

        # AIMessage object
        response = model_with_tools.invoke(input=complete_msg)


        return {
            'messages': [human_msg, response]
    }



def tool_node(state: dict):
    """Performs the tool call"""

    print("Entering tool node....\n")
    # print(f"Message state: {state['messages']}")
    
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))

    return {"messages": result}



def should_continue(state: AgentState) -> str:
    """Determine if we should continue or end the conversation."""

    messages = state["messages"]
    
    if not messages:
        return "continue"
    
    # This looks for the most recent tool message....
    for message in reversed(messages):
        # ... and checks if this is a ToolMessage resulting from save
        if (isinstance(message, ToolMessage) and 
            "saved" in message.content.lower() and
            "document" in message.content.lower()):
            return "end" # goes to the end edge which leads to the endpoint
        
    return "continue"


graph = StateGraph(AgentState)

graph.add_node("model_node", model_node)
graph.add_node("tool_node", tool_node)

graph.set_entry_point("model_node")

graph.add_edge("model_node", "tool_node")


graph.add_conditional_edges(
    "tool_node",
    should_continue,
    {
        "continue": "model_node",
        "end": END,
    },
)

app = graph.compile()


## Check agent flow ##
png_data = app.get_graph().draw_mermaid_png()

with open("graph.png", "wb") as f:
    f.write(png_data)




print("Graph saved as graph.png")

user_input = input("First message: ")
msg = {'messages': [HumanMessage(content=user_input)]}

app.invoke(msg)



