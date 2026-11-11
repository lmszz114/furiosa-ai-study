from kingkeras import run_once, get_cache_metrics

prefix = "NPU 추론 최적화에 대한 설명입니다." * 150

for i in range(1, 5):
    prompt = prefix + f"\n질문 {i}: KV Cache란?"

    print(f"\n===== {i}회 요청 =====")
    result = run_once(prompt, 100)

    print("응답 시간 : ", round(result["elapsed"], 3))
    print("입력 토큰 : ", result["input_tokens"])

    print("\nPrefix Cache 매트릭")
    get_cache_metrics()