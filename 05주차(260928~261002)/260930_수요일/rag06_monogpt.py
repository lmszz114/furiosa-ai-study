# rag05 카피

from langchain_openai import ChatOpenAI
import os

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1"
# 싸제 api 쓸거면 key 랑 url 둘다 필요

llm = ChatOpenAI(
    model_name='gpt-5.6-terra',
    temperature=0,
    # openai_api_key = openai_api_key,
    api_key=api_key,
    base_url=base_url,
)

response = llm.invoke('내가 누구게?')
print(response.content)