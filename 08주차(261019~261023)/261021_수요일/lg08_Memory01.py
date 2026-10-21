# lg07_2 카피

# tool calling : 외부 기능을 가져와서 사용하기

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

########################## 0. Tool 정의 ##########################
#pip install langchain-tavily
from langchain_tavily import TavilySearch

# 웹 검색 도구 생성
web_search = TavilySearch(
    max_results=3,
)
result = web_search.invoke({
    "query" : "삼성전자의 오늘 주가"
})

# print(result)
# print(result['results'][-1]['content'])

tools = [web_search]

llm_with_tools = llm.bind_tools(tools)

from typing import Annotated, TypedDict  # noqa: I001
from langgraph.graph import StateGraph, START, END  # noqa: F401
from langgraph.graph.message import add_messages  # noqa: F401
from langgraph.prebuilt import ToolNode, tools_condition

########################## 1. State 정의 ##########################
class State(TypedDict):
    messages : Annotated[list, add_messages]

########################## 2. Node 정의 ##########################
def chatbot(state : State) :
    return {"messages" : [llm.invoke(state["messages"])]}

################### 3. 그래프 구성, 노드 추가 ###################
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools))

################### 4. 엣지 추가 ###################
graph_builder.add_edge(START, "chatbot")
# graph_builder.add_edge("chatbot", "tools")
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition # tool_calls 가 있냐
    # tool_calls가 있으면 'tools' 반환
    # tool_calls가 없다면 END 반환
    # tools_condition에 END가 포함되어 있음
)
graph_builder.add_edge("tools", "chatbot")

################### 5. 컴파일 ###################
graph = graph_builder.compile()

################### 6. 그래프 실행 ###################

# 첫번째 질문
response = graph.invoke({
    "messages" : [("user", "내 이름은 윤영선이야.")]
})
print(response["messages"][-1].content)
print("==================== 두번째 질문 ====================")
# 두번째 질문
response = graph.invoke({
    "messages" : [("user", "내 이름이 뭐야.")]
})
print(response["messages"][-1].content)