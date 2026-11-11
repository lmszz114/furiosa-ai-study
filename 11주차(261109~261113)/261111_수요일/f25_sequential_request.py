import time
from kingkeras import get_llm

llm = get_llm(max_tokens=100)

prompts = [
    f"NPU 추론 최적호 방법 {i}번을 설명해줘"
    for i in range(8)
]

start = time.perf_counter()
for prompt in prompts:
    llm.invoke(prompt)
elapsed = time.perf_counter() - start

print("전체 요청:", len(prompts))
print("전체 소요시간: ", round(elapsed,3), "초")
print("요청 처리량:", round(len(prompts) / elapsed, 3), "req/s")