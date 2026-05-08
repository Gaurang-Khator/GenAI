from langchain_community.document_loaders import TextLoader

data = TextLoader(r"D:\Development\GenAI\RAG project\document loaders\notes.txt") #Creating object of TextLoader class 
# print(data)

docs = data.load()

print(docs[0].page_content)