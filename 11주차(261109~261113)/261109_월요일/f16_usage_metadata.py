from kingkeras import get_llm

llm = get_llm(max_tokens=500)

response = llm.invoke("NPU란 무엇인가")

print(response.content)
print("=================================")
print(response.usage_metadata)

