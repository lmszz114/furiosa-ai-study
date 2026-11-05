"""
furiosa-llm serve furiosa-ai/Qwen3-8B-FP8
furiosa-llm serve furiosa-ai/gpt-oss-20b
furiosa-llm serve furiosa-ai/Llama-3.1-8B-Instruct
furiosa-llm serve furiosa-ai/Meta-Llama-3.1-8B-Instruct-FP8-dynamic
furiosa-llm serve furiosa-ai/Qwen3-4B-FP8

furiosa-llm serve furiosa-ai/Qwen3-4B-FP8
     --devices npu:5 \
     --host 0.0.0.0 \
     --port 8000

"""

# from openai import OpenAI
from langchain_openai import ChatOpenAI

# models = client.models.list()
# model_id = models.data[0].id

# print("모델 : ", model_id)
llm = ChatOpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="EMPTY"
)

# 추론
# response = client.chat.completions.create(
#     model = model_id,
#     messages=[
#         {
#             "role" : "user",
#             "content" : "NPU와 GPU의 차이점을 한국어로 설명해줘."
#         }
#     ],
#     max_tokens=3000
# )

# print(response.choices[0].message.content)

response = llm.invoke("NPU와 GPU의 차이점을 설명해줘.")
print(response.content)