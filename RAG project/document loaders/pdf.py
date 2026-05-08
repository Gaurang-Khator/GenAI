from langchain_community.document_loaders import PyPDFLoader

data = PyPDFLoader(r"D:\Development\GenAI\RAG project\document loaders\cs-fundamentals.pdf") #Creating object of PyPDFLoader class
docs = data.load();

print(len(docs))