# rag01 카피

from langchain_openai import ChatOpenAI
import os
os.environ["OPENAI_API_KEY"] = 'test'
# 윈도우의 환경변수에 등록하는 방법 / 이 방법 역시 코드 상에 노출되므로 좋은 방법이 아님

llm = ChatOpenAI(
    model_name='gpt-5.6-terra',
    temperature=0,
    # openai_api_key = openai_api_key,
)

response = llm.invoke('내가 누구게?')
print(response.content)