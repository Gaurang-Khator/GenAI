from langchain_community.document_loaders import TextLoader

from langchain_text_splitters import CharacterTextSplitter 

splitter = CharacterTextSplitter(
    separator = "",
    chunk_size = 30,
    chunk_overlap = 1
)

data = TextLoader(r"D:\Development\GenAI\RAG project\document loaders\notes.txt") #Creating object of TextLoader class 
# print(data)

docs = data.load()

chunks = splitter.split_documents(docs)

# print(docs[0].page_content)

# print(len(chunks))

for i in chunks:
    print(i.page_content)
    print()
    print()