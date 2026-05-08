from dotenv import load_dotenv 
from langchain_mistralai import ChatMistralAI 
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate 

load_dotenv()

model = ChatMistralAI(model="mistral-small-2506")

data = TextLoader(r"D:\Development\GenAI\RAG project\document loaders\notes.txt")
docs = data.load()

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that summarizes the given document in 10 lines."),
    ("human", "{data}")
])

prompt = template.format_messages(data = docs[0].page_content)

response = model.invoke(prompt)

print(response.content)