# RAG AGENT

from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage, ToolMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict, Sequence
import os


llm = ChatMistralAI(model = "mistral-small-2506", temperature=0)


file_path = r"C:\Users\LENOVO\Downloads\Stock_Market_Performance_2024.pdf"

if not os.path.exists(file_path):
    raise FileNotFoundError(f"PDF file not found: {file_path}")

loader = PyPDFLoader(file_path)

try:
    pages = loader.load()
    print(f"Document has been loaded and it has {len(pages)} pages.")
except Exception as e:
    print(f"Error loading the document {str(e)}")
    raise 

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunks = splitter.split_documents(pages)

embedding_model = MistralAIEmbeddings(model = "mistral-embed")

persist_directory = r"D:\Development\GenAI\langgraph\stock_market"
collection_name = "stock_market"

if not os.path.exists(persist_directory):
    os.mkdir(persist_directory)

try:
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    print("ChromaDB vector store created successfully!")
except Exception as e:
    print(f"Error setting up ChromaDB {str(e)}")


retriever = vectorstore.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k" : 5}
)


@tool
def retriever_tool(query: str) -> str:
    """
     This tool searches and returns the information from the Stock Market Performance 2024 document.
    """

    docs = retriever.invoke(query)

    if not docs:
        return "I found no relevant information in the Stock Market Performance 2024 document."
    
    results = []

    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}: \n {doc.page_content}")

    return "\n\n".join(results)

tools = [retriever_tool]

llm = llm.bind_tools(tools)



class AgentState(TypedDict):
    messages : Annotated[Sequence[BaseMessage], add_messages]


def should_continue(state: AgentState):
    """Checks if the last message contains tool calls."""

    last_message = state['messages'][-1]

    return hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0


system_prompt = """
You are an intelligent AI assistant who answers questions about Stock Market Performance in 2024 based on the PDF document loaded into your knowledge base.
Use the retriever tool available to answer questions about the stock market performance data. You can make multiple calls if needed.
If you need to look up some information before asking a follow up question, you are allowed to do that!
Please always cite the specific parts of the documents you use in your answers.
"""

#LLM AGENT
def llm_agent(state: AgentState) -> AgentState:
    """Function to call llm with the current state."""

    all_messages = list(state['messages'] + [SystemMessage(content=system_prompt)])

    response = llm.invoke(all_messages)

    return {"messages" : [response]}

tools_dict = {t.name: t for t in tools}

#Retriever Agent
def retriever_agent(state: AgentState) -> AgentState:
    """Executes tool calls with llm's response."""

    results = []
    tool_calls = state['messages'][-1].tool_calls

    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")

        if not t['name'] in tools_dict:
            print(f"\n Tool: {t['name']} doesn't exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."

        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            print(f"Result length: {len(str(result))}")

        results.append(ToolMessage(tool_call_id=t['id'], tool_name=t['name'], content=str(result)))

    print("Tools execution complete.")
    
    return {"messages" : results}


graph = StateGraph(AgentState)

graph.add_node("agent", llm_agent)
graph.add_node("retriever", retriever_agent)

graph.set_entry_point("agent")

graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        True : "retriever",
        False : END
    }
)

graph.add_edge("retriever", "agent")

rag_agent = graph.compile()


def run_agent():
    print("\n=== RAG AGENT ===")

    while True:
        user_input = input("\n What is your question? ")

        if user_input.lower() == "exit" or user_input.lower() == "quit":
            break

        result = rag_agent.invoke({"messages" : HumanMessage(content=user_input)})

        print("\n === ANSWER === ")
        print(result['messages'][-1].content)

run_agent()
