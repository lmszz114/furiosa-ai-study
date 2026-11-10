# furiosa-llm serve furiosa-ai/Qwen3-8B-FP8 --devices npu:5 --host 127.0.0.1 --port 8000

import time
from langchain_openai import ChatOpenAI


def get_llm(max_tokens=None, temperature=None, top_p=None):

    llm = ChatOpenAI(
        base_url="http://127.0.0.1:8000/v1",
        api_key="EMPTY",
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p        
    )
    return llm

def run_once(promt, max_tokens=1000):
    llm = get_llm(max_tokens)

    start = time.perf_counter()
    response = llm.invoke(promt)
    elapsed = time.perf_counter() - start

    usage = response.usage_metadata or {}

    return {
        "elapsed" : elapsed,
        # "input_tokens" : usage['input_tokens'],
        "input_tokens" : usage.get("input_tokens", 0),
        "output_tokens" : usage.get("output_tokens", 0),
        "text" : response.content
    }

import urllib.request

def get_cache_metrics():
    url = "http://127.0.0.1:8000/metrics"
    with urllib.request.urlopen(url, timeout=5) as response:
        data = response.read().decode("utf-8")

    hits = 0
    queries = 0
    kv_usage = 0

    for line in data.splitlines():
        if line.startswith("#"):
            continue
        if "furiosa_llm_prefix_cache_hits{" in line:
            hits += float(line.split()[-1])
        elif "furiosa_llm_prefix_cache_queries{" in line:
            queries += float(line.split()[-1])
        elif "furiosa_llm_kv_cache_usage_percent{" in line:
            kv_usage = float(line.split()[-1]) * 100

    hit_rate = hits / queries * 100 if queries > 0 else 0

    print("\n===== Furiosa NPU Cache 상태 =====")
    print(f"Prefix Cache 조회 횟수 : {queries:.0f}")
    print(f"Prefix Cache 적중 횟수 : {hits:.0f}")
    print(f"Prefix Cache 적중률 : {hit_rate:.2f}")
    print(f"KV Cache 사용률 : {kv_usage:.2f}")

    return{
        "queries" : queries,
        "hits" : hits,
        "hit_rate" : hit_rate,
        "kv_usage" : kv_usage
    }


# get_cache_metrics()