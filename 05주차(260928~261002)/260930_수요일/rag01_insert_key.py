from langchain_openai import ChatOpenAI

openai_api_key = 'test'
# 실무에선 키 자체를 넣는 일은 없으나, 지금은 실습으로 테스트를 위해 임시로 진행

llm = ChatOpenAI(
    model_name='gpt-5.6-terra',
    temperature=0,
    openai_api_key = openai_api_key,
)

response = llm.invoke('내가 누구게?')   # invoke 한번 하고나면 휘발됨 (내용이 메모리에 저장되지 않음)
print(response.content)