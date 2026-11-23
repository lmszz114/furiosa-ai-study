from kingkeras import get_llm
from langchain_core.tools import tool
from langchain.agents import create_agent
# pip install langchain
from langchain_core.messages import AIMessage

@tool
def add(a:int, b:int) -> int:
    """두 정수를 더합니다."""
    return a + b

@tool
def subtract(a:int, b:int) -> int:  # 타입힌트
    """첫번째 정수에서 두번째 정수를 뺍니다."""
    return a - b

@tool
def multiply(a:int, b:int) -> int:  # 타입힌트
    """두 정수를 곱합니다."""
    return a * b

@tool
def divide(a:int, b:int) -> float:  # 타입힌트
    """첫번째 정수를 두번째 정수로 나눕니다."""
    return a / b

llm = get_llm(max_tokens=500)
tools = [add, multiply, subtract, divide]

agent = create_agent(
    model = llm,
    tools = tools,
    system_prompt="계산이 필요하면 도구를 사용하세요.",
    # 에이전트한테 역할과 행동지침을 설정
)

questions = [
    "100에서 35를 빼줘",
    "12와 8을 곱해줘",
    "100을 4로 나눠줘",
    "15와 27을 더해줘"
]

for question in questions:
    print("\n질문: ",question)
    result = agent.invoke({
        "messages": [
            {"role":"user", "content":question}
        ]
    })

# print("====================================")
# # print(result)
# print("====================================")
# for i, msg in enumerate(result["messages"]):
#     print(f"{i+1}번째 메시지")
#     print("타입: ", type(msg).__name__)
#     # print("내용: ", msg.content)

#     if isinstance(msg, AIMessage) and msg.tool_calls:
#         for call in msg.tool_calls:
#             print("도구 이름: ", call['name'])
#             print("전달 인자: ", call['args'])
    print("====================================")
    print("최종 답변: ", result["messages"][-1].content)