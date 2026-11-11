import time
from concurrent.futures import ThreadPoolExecutor
from kingkeras import run_once

def ask(name, delay, max_tokens):
    time.sleep(delay)

    print(f"{name} 요청 시작", flush=True)

    start = time.perf_counter()
    result = run_once(
        f"NPU와 LLM 추론 최적화를 자세히 설명해줘. 요청 {name}",
        max_tokens
    )
    elapsed = time.perf_counter() - start
    
    print(f"{name} 요청 완료: {elapsed:.2f}초", flush=True)

    return result

with ThreadPoolExecutor(max_workers=4) as pool:
    futures = [
        pool.submit(ask, "A", 0, 800),
        pool.submit(ask, "B", 1, 200),
        pool.submit(ask, "C", 2, 1000),
        pool.submit(ask, "D", 3, 200),
        pool.submit(ask, "E", 4, 300),
    ]
    results = [f.result() for f in futures]

print("=========== 끝 ============")