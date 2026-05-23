from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI

from langchain_classic.retrievers.multi_query import MultiQueryRetriever

docs = [
    Document(page_content="Gradient Descent is an optimization algorithm used in machine learning"),
    Document(page_content="Gradient Descent minimizes the loss function."),
    Document(page_content="Gradient Descent is an optimiztion technique that minimizes the loss function"),
    Document(page_content="Neural networks uses gradient descent for training"),
    Document(page_content="Support vector machines are supervised learning algorithms.")
]

llm = ChatMistralAI(model="mistral-small-2506")

embedding_model = MistralAIEmbeddings(model="mistral-embed")

vectorstore = Chroma.from_documents(docs, embedding_model)

retriever = vectorstore.as_retriever()

multi_query_retriever = MultiQueryRetriever.from_llm(
    llm = llm,
    retriever = retriever
)

query = "What is gradient descent?"

docs = multi_query_retriever.invoke(query) 

print("\n====MULTI QUERY RETRIEVER RESULTS====")

for doc in docs:
    print(doc.page_content)