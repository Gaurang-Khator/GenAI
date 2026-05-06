from dotenv import load_dotenv 
from langchain_mistralai import ChatMistralAI 

load_dotenv()

model = ChatMistralAI(model="mistral-small-2506")

response = model.invoke("Hello!")

print(response.content)