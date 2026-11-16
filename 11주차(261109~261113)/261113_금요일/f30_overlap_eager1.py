# furiosa-llm serve furiosa-ai/Qwen3-8B-FP8 --devices npu:5 --host 127.0.0.1 --port 8000
# furiosa-llm serve furiosa-ai/Qwen3-8B-FP8 --devices npu:5 --host 127.0.0.1 --port 8000 --enable-overlap-schedulin

import time
from kingkeras import get_llm

llm = get_llm(max_tokens=100)

prompts = [
    f"NPU 추론 최적화 방법 {i}번을 설명해줘."
    for i in range(8)
]

start = time.perf_counter()
total_tokens = 0

for prompt in prompts:
    response = llm.invoke(prompt)

    usage = response.usage_metadata or {}
    tokens = usage.get("output_tokens", 0)

    total_tokens += tokens
elapsed = time.perf_counter() - start

print("전체 요청 : ", len(prompts))
print("전체 출력 토큰 : ", total_tokens)
print("전체 소요시간 : ", round(elapsed,3), "초")
print("RPS : ", round(len(prompts) / elapsed, 3))
print("TPS : ", round(total_tokens / elapsed, 3))


######################## eager mode ########################
# 전체 요청 :  8
# 전체 출력 토큰 :  800
# 전체 소요시간 :  8.805 초
# RPS :  0.909
# TPS :  90.86

######################## overlap mode ########################
# 전체 요청 :  8
# 전체 출력 토큰 :  800
# 전체 소요시간 :  8.721 초
# RPS :  0.917
# TPS :  91.733