from dotenv import load_dotenv
load_dotenv()

from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.documents import Document

docs = [
    Document(page_content="My name is Gaurang Khator and I am a recent B.Tech graduate from VIT in the branch of Computer Science and Engineering.", metadata={"source": "about.txt"}),
    Document(page_content="I am AWS Cloud Practitioner certified and have keen interest in field of Generative AI and Agentic AI.", metadata={"source":"info.txt"}),
    Document(page_content="My skills include Python, C++, SQL, AWS, Langchain, RAG, Nextjs, Reactjs, Nodejs, Git and Github.", metadata={"Source":"skill.txt"})
]

embedding_model = MistralAIEmbeddings( model = "mistral-embed" )

vector_store = Chroma.from_documents(
    documents = docs,
    embedding = embedding_model,
    persist_directory = "chroma_db"
)

result = vector_store.similarity_search("What skills do Gaurang brings to the table?", k=2)

for r in result:
    print(r.page_content)
    print(r.metadata)
