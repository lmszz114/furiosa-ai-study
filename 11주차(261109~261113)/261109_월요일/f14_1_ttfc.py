# time to first chunk

from kingkeras import get_llm
import time

llm = get_llm()

prompt = "NPU의 장점을 자세히 설명해줘"

start = time.perf_counter()
first_time = None

for chunk in llm.stream(prompt):

    if chunk.content:
        if first_time is None:
            first_time = time.perf_counter() - start
        print(chunk.content, end="", flush=True)

print("\n==============================================")
print("첫 컨텐츠 응답시간 : ", round(first_time,2), "초")