#load pdf
#split into chunks
#create embeddings
#store in chromadb

from langchain_community.document_loaders import PyPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv 

load_dotenv()

data = PyPDFLoader(r"D:\Development\GenAI\RAG project\document loaders\deeplearning.pdf")       #Creating object of PyPDFLoader class
docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunks = splitter.split_documents(docs)


embedding_model = MistralAIEmbeddings( model = "mistral-embed" )

vector_store = Chroma.from_documents(
    documents = chunks,
    embedding = embedding_model,
    persist_directory = "chroma_db"
)