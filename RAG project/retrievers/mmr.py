#Similarity search and MMR (Maximal Marginal Relevance) search 

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

docs = [
    Document(page_content="Gradient Descent is an optimization algorithm used in machine learning"),
    Document(page_content="Gradient Descent minimizes the loss function."),
    Document(page_content="Gradient Descent is an optimiztion technique that minimizes the loss function"),
    Document(page_content="Neural networks uses gradient descent for training"),
    Document(page_content="Support vector machines are supervised learning algorithms.")
]

embedding_model = MistralAIEmbeddings(model="mistral-embed")

vector_store = Chroma.from_documents(docs, embedding_model)

#Similarity search
similarity_search = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

print("\n====SIMILARITY SEARCH RESULTS====")

similarity_docs = similarity_search.invoke("What is gradient descent?")

for doc in similarity_docs:
    print(doc.page_content)

#MMR search
mmr_search = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3}
)

print("\n====MMR SEARCH RESULTS====")

mmr_docs = mmr_search.invoke("What is gradient descent?")

for doc in mmr_docs:
    print(doc.page_content)