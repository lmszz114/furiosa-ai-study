from kingkeras import run_once, get_cache_metrics

prompts = [
    "NPU란 무엇인가?",
    "NPU의 특징을 설명해줘." * 100,
    "NPU의 특징을 설명해줘." * 300,
]

for i, prompt in enumerate(prompts, 1):
    print(f"\n========== 실험 {i} ==========")
    result = run_once(prompt, 300)
    print("입력 토큰 : ", result["input_tokens"])
    print("출력 토큰 : ", result["output_tokens"])
    print("응답 시간 : ", round(result["elapsed"],2))

    print("\nKV Cache 매트릭")
    get_cache_metrics()