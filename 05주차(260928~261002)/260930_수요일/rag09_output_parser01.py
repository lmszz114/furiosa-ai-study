# rag08_01 카피

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

import os
from dotenv import load_dotenv
load_dotenv()

########################### API ###########################
api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1"
model = ChatOpenAI(
    model_name='gpt-5.6-terra',
    temperature=0,
    # openai_api_key = openai_api_key,
    api_key=api_key,
    base_url=base_url,
)
###########################################################
from langchain_core.output_parsers import StrOutputParser
output_parser = StrOutputParser()

prompt = PromptTemplate.from_template("{topic}에 대해 쉽게 설명해줘.")

chain = prompt | model | output_parser

input = {"topic" : "양자컴퓨터 학습 원리"}

response = chain.invoke(input)
print(response)