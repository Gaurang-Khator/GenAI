from langchain_community.document_loaders import PyPDFLoader

data = PyPDFLoader(r"D:\Development\GenAI\RAG project\document loaders\cs-fundamentals.pdf")
docs = data.load();

print(len(docs))