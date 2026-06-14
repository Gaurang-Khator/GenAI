from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langgraph.graph import START, END, StateGraph
from langchain_core.messages import SystemMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from typing import TypedDict, Sequence, Annotated
from langgraph.prebuilt import ToolNode


class AgentState(TypedDict):
    messages : Annotated[Sequence[BaseMessage], add_messages]

@tool
def add(a: int, b: int):
    """This is an addition functionn that adds two numbers."""

    return a+b

@tool
def subtract(a: int, b: int):
    """This is an subtraction functionn."""

    return a-b

@tool
def multiply(a: int, b: int):
    """This is an multiplication functionn."""

    return a*b

tools = [add, subtract, multiply]

model = ChatMistralAI(model = "mistral-small-2506")

model = model.bind_tools(tools)

def model_call(state: AgentState) -> AgentState:
    
    system_prompt = SystemMessage(content="You are my helpful AI assistant. Answer my query to the best of your ability.")

    response = model.invoke([system_prompt] + state['messages'])

    return {"messages" : [response]} #not replaces due to add_messages reducer function
    # state['messages'] = response
    # return state


def should_continue(state: AgentState):
    """This is a decider function which decides will the loop in the graph continue or ends."""

    messages = state['messages']
    last_message = messages[-1]

    if not last_message.tool_calls:
        return "end_edge"
    else: 
        return "continue_edge"
    


graph = StateGraph(AgentState)

graph.add_node("agent", model_call)
graph.set_entry_point("agent")

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        #edge : node
        "continue_edge" : "tools",
        "end_edge" : END        
    }
)

graph.add_edge("tools", "agent")

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages" : [("user", "Add 12 + 5. Then Subtract result - 1. Then multiply the result by 4. also tell me a joke.")]} 
print_stream(app.stream(inputs, stream_mode="values"))