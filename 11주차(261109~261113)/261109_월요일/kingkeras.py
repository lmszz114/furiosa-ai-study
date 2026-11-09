# furiosa-llm serve furiosa-ai/Qwen3-8B-FP8 --devices npu:5 --host 127.0.0.1 --port 8000

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