# rag09_01 카피

template = """
당신은 영어를 가르치는 10년차 영어 선생님입니다.
주어진 상황에 맞는 영어 회화에 맞는 영어회화를 작성해주세요.
양식은 [FORMAT]을 참고하여 작성해주세요.

# 상황
{question}

# FORMAT
- 영어회화:
- 한글번역:
"""

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

prompt = PromptTemplate.from_template(template=template)

chain = prompt | model | output_parser

input = {"question" : "강남역에서 판교역까지 가장 빠르게 갈 수 있는 방법이 궁금해요."}

response = chain.invoke(input)
print(response)