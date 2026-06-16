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


file_path = r""

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

persist_directory = r"D:\Development\GenAI\langgraph"
collection_name = "stock_market"

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
