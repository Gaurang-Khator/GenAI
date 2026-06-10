# Simple BOT

from typing import TypedDict, List
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages : List[HumanMessage]

llm = ChatMistralAI(model = "mistral-small-2506")

def process_node(state: AgentState) -> AgentState:
    """This is a simple node."""

    response = llm.invoke(state['messages'])
    print("\nAI: ", response.content)
    return state

graph = StateGraph(AgentState)

graph.add_node("process", process_node)

graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()

user_input = input("Enter: ")
while(user_input != "exit"):
    agent.invoke({"messages" : [HumanMessage(content=user_input)]})
    user_input = input("Enter: ")