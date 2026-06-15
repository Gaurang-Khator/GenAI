# MINI PROJECT - DRAFTER

from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, BaseMessage, ToolMessage
from typing import TypedDict, Annotated, Sequence


document_content = ""

class AgentState(TypedDict):
    messages : Annotated[Sequence[BaseMessage], add_messages]

@tool
def update_tool(content: str) -> str:
    """Updates the content with the provided content."""

    global document_content
    document_content = content
    return f"Document updated successfully! Updated(current) content is: \n{document_content}"

@tool
def save_tool(filename: str) -> str:
    """Save the current document to a text file and finish the process.
    
    Args: 
        filename : The name of the text file.
    """

    global document_content

    if not filename.endswith(".txt"):
        filename = f"{filename}.txt"

    try:
        with open(filename, 'w') as file:
            file.write(document_content)
        
        print(f"\nDocument has been saved to: '{filename}'")
        return f"The document has been saved successfully to '{filename}'."
    
    except Exception as e:
        return f"Error saving file: \n{str(e)}"
    

tools = [update_tool, save_tool]

llm = ChatMistralAI(model = "mistral-small-2506").bind_tools(tools)

def our_agent(state: AgentState) -> AgentState:
    
    system_prompt = SystemMessage(content=f"""
    You are a Drafter, a helpful writing assistant. You are going to help user update and modify documents.
    
    - If the user wants to update or modify content, use the 'update_tool' tool with the complete updated content.
    - If the user wants to save and finish, use the 'save_tool' tool.
    - Make sure to always show the current document state after modifications.
                                  
    The current document content is: {document_content}
    """)

    if not state['messages']:
        user_input = "I'm ready to help you update a document. What would you like to create?"
        user_message = HumanMessage(content=user_input)

    else:
        user_input = input("\nWhat would you like to do with the document? ")
        print(f"\n USER: {user_input}")
        user_message = HumanMessage(content=user_input)

    all_messages = [system_prompt] + list(state['messages']) + [user_message]
    
    response = llm.invoke(all_messages)

    print(f"\n AI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f" Using TOOLS: {[tc['name'] for tc in response.tool_calls]}")

    return {"messages" : list(state['messages']) + [user_message, response]}


def should_continue(state: AgentState) -> str:
    """Determines whether we should continue or end the conversation."""

    messages = state['messages']

    if not messages:
        return "continue_edge"
    
    for m in reversed(messages):
        if(isinstance(m, ToolMessage) and "saved" in m.content.lower() and "document" in m.content.lower()):
            return "end_edge"
        
    return "continue_edge"


def print_messages(messages):
    """Function to print the messages in a more readable format"""
    if not messages:
        return
    
    for m in messages[-3:]:
        if isinstance(m, ToolMessage):
            print(f"\n TOOL RESULT: {m.content}")



graph = StateGraph(AgentState)

graph.add_node("agent", our_agent)
graph.add_node("tools", ToolNode(tools))

graph.set_entry_point("agent")

graph.add_edge("agent", "tools")

graph.add_conditional_edges(
    "tools", 
    should_continue,
    {
        "continue_edge" : "agent",
        "end_edge" : END
    }
)

app = graph.compile()


def run_document_agent():

    print("\n === DRAFTER ===")

    state = {"messages" : []}

    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(state['messages'])

    print("\n === DRAFTER FINISHED ===")

if __name__ == "__main__":
    run_document_agent()