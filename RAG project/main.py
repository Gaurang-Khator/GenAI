from dotenv import load_dotenv 
from langchain_mistralai import ChatMistralAI 
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate 
from langchain_text_splitters import RecursiveCharacterTextSplitter 

load_dotenv()

model = ChatMistralAI(model="mistral-small-2506")

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

# data = TextLoader(r"D:\Development\GenAI\RAG project\document loaders\notes.txt")                #Creating object of TextLoader class
data = PyPDFLoader(r"D:\Development\GenAI\RAG project\document loaders\deeplearning.pdf")       #Creating object of PyPDFLoader class
docs = data.load()

chunks = splitter.split_documents(docs)

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that summarizes the given document in 10 lines."),
    ("human", "{data}")
])

prompt = template.format_messages(data = docs[4].page_content)

response = model.invoke(prompt)

print(response.content)