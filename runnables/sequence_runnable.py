from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_mistralai import ChatMistralAI

#1. Prompt
prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in simple terms."
)

#2. LLM Model
llm_model = ChatMistralAI(model="mistral-small-2506")

#3. Output Parser
parser = StrOutputParser()


#Step-by-Step Manual flow

# #format the prompt with the topic
# formatted_prompt = prompt.format_messages(topic="deep learning")

# #call the model manually
# response = llm_model.invoke(formatted_prompt)

# #parse the output manually
# output = parser.parse(response.content)

# print(output)


# SEQUENCE RUNNABLE

chain = prompt | llm_model | parser

result = chain.invoke("deep learning")
print(result)