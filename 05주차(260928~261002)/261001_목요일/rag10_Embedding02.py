# rag10_01 카피

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

import os
from dotenv import load_dotenv
load_dotenv()

########################### API ###########################
api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1"
# model = ChatOpenAI(
#     model_name='gpt-5.6-terra',
#     temperature=0,
#     # openai_api_key = openai_api_key,
#     api_key=api_key,
#     base_url=base_url,
# )
###########################################################

prompt = "삼성전자의 창업주는 누구인가요?"

from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
    model = "text-embedding-3-small",
    api_key=api_key,
    base_url=base_url,
    dimensions=5,   # 차원 조절 가능
)

vector = embeddings.embed_query(prompt)
print(vector)
print("=======================================")
print("임베딩 벡터의 차원 :", len(vector))
# 임베딩 벡터의 차원 : 1536