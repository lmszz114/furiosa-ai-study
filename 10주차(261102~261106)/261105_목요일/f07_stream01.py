from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="EMPTY"
)

model = client.models.list().data[0].id

# 추론
streaming = client.chat.completions.create(
    model = model,
    messages=[
        {
            "role" : "user",
            "content" : "NPU와 GPU의 차이점을 한국어로 설명해줘."
        }
    ],
    max_tokens=300,
    stream=True
)

for chunk in streaming:
    if chunk.choices:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", flush=True)

