# ChatBot with Memory

from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, START, END
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import AIMessage, HumanMessage
from typing import TypedDict, List, Union
import os

class AgentState(TypedDict):
    messages : List[Union[HumanMessage, AIMessage]]

llm = ChatMistralAI(model = "mistral-small-2506")

def process_node(state: AgentState) -> AgentState:
    """This is a simple node."""

    response = llm.invoke(state['messages'])

    state['messages'].append(AIMessage(content=response.content))
    print(f"\nAI: {response.content}")
    print(f"CURRENT STATE: {state['messages']}")

    return state

graph = StateGraph(AgentState)
graph.add_node("process", process_node)
graph.add_edge(START, "process")
graph.add_edge("process", END)
app = graph.compile()


user_input = input("Enter: ")

conversation_history = []

while user_input != "exit":

    conversation_history.append(HumanMessage(content=user_input))

    result = app.invoke({"messages" : conversation_history})

    conversation_history = result['messages']

    user_input = input("Enter: ")


with open("logging.txt", "w", encoding="utf-8") as file:
    file.write("Conversation Log:\n")

    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"You: {message.content}\n")
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n")

    file.write("\n End of Conversation \n")