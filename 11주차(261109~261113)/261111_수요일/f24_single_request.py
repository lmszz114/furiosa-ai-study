import time
from kingkeras import get_llm

llm = get_llm(max_tokens=100)

start = time.perf_counter()
resp = llm.invoke("NPU의 장점 3가지를 설명해줘.")
elapsed = time.perf_counter() - start

usage = resp.usage_metadata or {}

print("응답:", resp.content)
print("지연 시간: ", round(elapsed,3))
print("input:", usage.get("input_tokens",0))
print("output:", usage.get("output_tokens",0))