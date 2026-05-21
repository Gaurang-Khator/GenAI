from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

data = PyPDFLoader(r"D:\Development\GenAI\RAG project\document loaders\cs-fundamentals.pdf") #Creating object of PyPDFLoader class
docs = data.load();

chunks = splitter.split_documents(docs)

print(len(chunks))