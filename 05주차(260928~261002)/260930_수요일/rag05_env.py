# rag03 카피

from langchain_openai import ChatOpenAI
import os

from dotenv import load_dotenv
load_dotenv()
# vscode에 한해서 dotenv 생략해도 돌아가지만, 다른 환경에선 안되기 때문에 기본적으론 써줘야 함 

llm = ChatOpenAI(
    model_name='gpt-5.6-terra',
    temperature=0,
    # openai_api_key = openai_api_key,
)

response = llm.invoke('내가 누구게?')
print(response.content)