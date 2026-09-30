# rag02 카피

from langchain_openai import ChatOpenAI
import os
# os.environ["OPENAI_API_KEY"] = 'test'
# 시스템 환경 변수에 직접 등록하고 윗줄 제거

llm = ChatOpenAI(
    model_name='gpt-5.6-terra',
    temperature=0,
    # openai_api_key = openai_api_key,
)

response = llm.invoke('내가 누구게?')
print(response.content)