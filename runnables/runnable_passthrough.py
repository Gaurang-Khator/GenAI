from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough


llm = ChatMistralAI(model="mistral-small-2506")
parser = StrOutputParser()


code_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a code generator"),
    ("human", "{topic}")
])

explain_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant who explains the code in detail."),
    ("human", "Explain the following codde in simple terms:\n {code}")
])

seq = code_prompt | llm | parser

seq2 = RunnableParallel({
    "code": RunnablePassthrough(),
    "explanation": explain_prompt | llm | parser
})

chain = seq | seq2

result = chain.invoke({"topic": "Write a code of binary search in C++."})

print(result['code'])
print(result['explanation'])