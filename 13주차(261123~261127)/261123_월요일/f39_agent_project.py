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
    system_prompt=(
        "넌 계산 전문 에이전트야."
        "계산이 필요하면 등록된 도구를 사용해."
        "여러 단계의 계산은 순서대로 수행해."
        "계산 오류가 발생하면 사용자에게 설명해."
    )
)

# Agent 함수 만들기
def run_agent(question):
    print("\n사용자 질문 : ", question)
    print("====================================")

    result = agent.invoke(
        {
            "messages": [
                {"role":"user", "content":question}
            ]
        },
        config={"recursion_limit":20}
    )

    for i, msg in enumerate(result["messages"]):

        if isinstance(msg, AIMessage) and msg.tool_calls:
            for call in msg.tool_calls:
                print("도구 이름: ", call['name'])
                print("전달 인자: ", call['args'])

    print("====================================")
    print("최종 답변: ", result["messages"][-1].content)

    return result

# 사용자 입력
while True:
    question = input("\n계산할 질문을 입력하세요 (종료:q):")

    if question.lower() == 'q':
        print("Agent 종료")
        break

    if not question.strip():
        continue

    try:
        run_agent(question)
    # except ValueError as e:
    #     print("계산오류: ", e)
    except Exception as e:
        print("Agent 실행 오류: ", type(e).__name__, e)