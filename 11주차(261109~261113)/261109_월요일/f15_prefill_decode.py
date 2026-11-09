from kingkeras import get_llm
import time

llm = get_llm(max_tokens=2000)

prompts = [
    "NPU란?",
    "NPU와 GPU의 차이를 10가지 설명해줘.",
    "NPU의 구조와 장점, 활용 분야를 더더더 자세히 설명해줘."
]

for prompt in prompts:
    start = time.perf_counter()
    response = llm.invoke(prompt)
    elapsed = time.perf_counter() - start

    print("질문 : ", prompt)
    print("시간 : ", round(elapsed, 2), "초")