chave_openai = "insira sua chave da api open ai"

#pip install langchain

#pip install openai

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

from langchain.schema import (
    HumanMessage,
    SystemMessage
)

llm = ChatOpenAI(openai_api_key=chave_openai, temperature=0.9, model_name="gpt-3.5-turbo")

response = llm([HumanMessage(content="Qual a capital do Brasil?")])

print(response)