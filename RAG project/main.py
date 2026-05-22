from dotenv import load_dotenv 
from langchain_mistralai import ChatMistralAI 
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate 
from langchain_text_splitters import RecursiveCharacterTextSplitter 

load_dotenv()

model = ChatMistralAI(model="mistral-small-2506")


template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that summarizes the given document in 10 lines."),
    ("human", "{data}")
])

