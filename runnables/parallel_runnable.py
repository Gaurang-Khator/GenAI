from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda

llm = ChatMistralAI(model = "mistral-small-2506")
parser = StrOutputParser()

short_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in 1-2 lines."
)

detailed_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in detail."
)

# MANUALLY

# formatted_short = short_prompt.format_messages(topic="machine learning")
# response_short = llm.invoke(formatted_short)
# output_short = parser.parse(response_short.content)
# print("Short Explanation:", output_short)

# PARALLEL RUNNABLE

# Same input for both prompts

# runnable = RunnableParallel({
#     "short" : short_prompt | llm | parser,
#     "detailed" : detailed_prompt | llm | parser
# })

# result = runnable.invoke({"topic": "machine learning"})
# print(result['short'])
# print(result['detailed'])

# Different input for each prompt

runnable = RunnableParallel({
    "short" : RunnableLambda(lambda x: x['short']) | short_prompt | llm | parser,
    "detailed" : RunnableLambda(lambda x: x['detailed']) | detailed_prompt | llm | parser
})

result = runnable.invoke({
    "short": {"topic": "machine learning"},
    "detailed": {"topic": "deep learning"}
})

print(result['short'])
print(result['detailed'])