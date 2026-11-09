from kingkeras import get_llm
import time

llm = get_llm(max_tokens=500)

prompt = "NPU의 특징을 말해줘."

times = []
for i in range(5):
    start = time.perf_counter()

    response = llm.invoke(prompt)

    end = time.perf_counter()
    elasped = end - start

    times.append(elasped)
    print(i + 1, "회: ", round(elasped, 2), "초")
average = sum(times) / len(times)

print("평균: ", round(average, 2), "초")