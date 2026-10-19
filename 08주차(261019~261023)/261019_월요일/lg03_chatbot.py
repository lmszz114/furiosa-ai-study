from langchain_openai import ChatOpenAI  # noqa: I001
import os

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1"

llm = ChatOpenAI(
    model_name='gpt-5.6-terra',
    temperature=0,
    max_tokens=1000,
    # openai_api_key = openai_api_key,
    api_key=api_key,
    base_url=base_url,
)

from typing import Annotated, TypedDict  # noqa: F401, I001
from langgraph.graph import StateGraph, START, END  # noqa: F401
from langgraph.graph.message import add_messages  # noqa: F401

################### 1. State 정의 ###################
class State(TypedDict):
    messages : Annotated[list, add_messages]
    aaa : str
    bbb : str
    ccc : int
# 위에꺼는 한마디로 아래임
# state = {
#     "messages" : [],
#     'aaa' : 'abc'
# }

################### 2. Node 정의 ###################
def chatbot(state : State) :
    return {"messages" : [llm.invoke(state["messages"])]}

################### 3. 그래프 구성 ###################
graph_01 = StateGraph(State)
graph_01.add_node("chatbot", chatbot)

################### 4. 엣지 추가 ###################
graph_01.add_edge(START, "chatbot")
graph_01.add_edge("chatbot", END)

################### 5. 그래프 컴파일 ###################
graph = graph_01.compile()

################### 6. 그래프 실행 ###################
query = "삼성전자의 주가를 예측해줘"

response = graph.invoke({
    "messages" : [("user", query)]
})

print(response['messages'][1].content)
print(response['messages'][0].type)
print(response['messages'][1].type)
