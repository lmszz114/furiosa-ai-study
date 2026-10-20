# 261020 복습 — LangGraph 도구 호출(Tool Calling): ToolNode · 조건부 엣지

> 어제는 START에서 END까지 한 줄로 흐르는 직선 그래프를 만들었다. 오늘은 LLM이 **외부 도구(웹 검색)를 가져다 쓰게** 하고, "도구가 필요하면 도구로, 아니면 끝으로" 갈라지는 **조건부 엣지**로 그래프가 상황에 따라 분기·반복하도록 만든다.

## 목차
1. 오늘의 핵심 — 도구 호출(Tool Calling)이란
2. lg05_ToolNode — 웹 검색 도구 붙이기 (Tavily) + 조건부 엣지
3. lg06 — 검색 도구 바꾸기·추가하기 (DuckDuckGo)
4. lg07 — 그래프를 그림으로 저장하기
5. 한 줄 요약

---

## 1. 오늘의 핵심 — 도구 호출(Tool Calling)이란

LLM은 학습한 시점까지의 지식만 알고, "오늘 삼성전자 주가" 같은 **실시간 정보나 최신 사실은 모른다.** 이럴 때 LLM에게 **외부 도구(웹 검색, 계산기, DB 조회 등)를 쥐어주고, 필요하면 그 도구를 쓰게** 하는 것이 도구 호출(Tool Calling)이다.

**비유하자면, 시험 보는 사람에게 "모르는 건 인터넷을 검색해서 답하라"고 허락하는 것이다.** 아는 질문은 그냥 답하고, 모르는 질문이 나오면 검색 도구를 꺼내 쓴 뒤 그 결과로 답한다.

동작의 큰 그림은 이렇다.

```
사용자 질문 → [LLM] "이건 내가 아는 거야" → 바로 답변 → 끝
                 "이건 실시간 정보라 몰라" → 검색 도구 호출 → [도구 실행] → 결과를 받아 → [LLM] 답변 → 끝
```

LangGraph에서는 이 흐름을 **노드 두 개(LLM 노드, 도구 노드)와 조건부 엣지**로 만든다. LLM 노드가 "도구가 필요한지"를 판단하고, 조건부 엣지가 그 판단에 따라 도구 노드로 보낼지 끝낼지를 정한다.

---

## 2. lg05_ToolNode — 웹 검색 도구 붙이기 (Tavily) + 조건부 엣지

### 2-1. 도구 준비 — Tavily 웹 검색

**Tavily**는 AI 앱용 웹 검색 API다. 쓰려면 사전 준비가 필요하다.

```bash
pip install langchain-tavily
```
- `app.tavily.com`에서 API 키를 발급받아, `.env` 파일에 `TAVILY_API_KEY = "..."`로 넣어둔다. (API 키는 그 서비스를 쓸 자격을 증명하는 열쇠로, 코드에 직접 쓰지 않고 `.env`에 보관한다.)

```python
from langchain_tavily import TavilySearch

web_search = TavilySearch(max_results=3)   # 검색 결과 3개까지

# 도구 단독 테스트 — 그래프와 무관하게 도구만 호출해 보기
result = web_search.invoke({"query" : "삼성전자의 오늘 주가"})
```

- **`TavilySearch(...)`** 가 웹 검색 도구 하나다. `max_results`로 가져올 결과 수를 정한다. `invoke`로 직접 호출하면 검색 결과가 그대로 돌아온다(그래프에 넣기 전, 도구가 잘 되는지 확인하는 용도).

### 2-2. LLM에 도구 쥐어주기 — bind_tools

```python
tools = [web_search]
llm_with_tools = llm.bind_tools(tools)
```

- **`bind_tools(tools)`** 는 LLM에게 "너는 이런 도구들을 쓸 수 있다"고 **도구 목록을 쥐어주는** 것이다. 이렇게 묶인 LLM은 질문을 받으면 "내가 직접 답할지, 아니면 도구를 써야 할지"를 판단하고, 도구가 필요하면 "이 도구를 이런 검색어로 써줘"라는 **도구 호출 요청(tool_calls)** 을 만들어 낸다.

### 2-3. 도구 실행 노드와 조건부 엣지 — ToolNode · tools_condition

```python
from langgraph.prebuilt import ToolNode, tools_condition

# 노드 두 개
graph_builder.add_node("chatbot", chatbot)       # LLM 노드
graph_builder.add_node("tools", ToolNode(tools)) # 도구 실행 노드

# 엣지
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,   # chatbot 뒤에서 조건에 따라 갈림
    # 도구 호출(tool_calls)이 있으면 → "tools" 로
    # 없으면 → END 로 (tools_condition 안에 END가 포함돼 있음)
)
graph_builder.add_edge("tools", "chatbot")   # 도구 실행 후 다시 chatbot 으로
```

- **`ToolNode(tools)`** 는 LLM이 요청한 도구를 **실제로 실행하는 노드**다. LLM이 "이 검색어로 웹 검색해줘"라고 요청하면, 이 노드가 진짜 검색을 수행하고 결과를 상태(messages)에 담는다.
- **`tools_condition`** 은 LangGraph가 미리 만들어 둔 **조건 판단 함수(조건부 엣지)** 다. 직전 LLM의 답에 도구 호출 요청이 있는지를 보고, **있으면 `"tools"` 노드로, 없으면 `END`로** 보낸다. (어제 배운 `add_conditional_edges`가 바로 이 "상태를 보고 갈라지는 엣지"다.)
- **`add_edge("tools", "chatbot")` — 도구 실행 후 다시 LLM으로 되돌아온다.** 이게 순환(루프)을 만든다.

**전체 흐름(순환 구조).**
```
[START] → (chatbot) ──도구 필요?──┬─ 있음 → (tools: 검색 실행) ─┐
                                   │                             │ 결과를 들고
                                   └─ 없음 → [END]               └──→ 다시 (chatbot)
```
"삼성전자 오늘 주가"를 물으면: ① chatbot(LLM)이 "실시간 정보라 검색이 필요"하다고 판단해 도구 호출을 만들고 → ② tools_condition이 "tools"로 보내 → ③ 웹 검색이 실행되고 → ④ 결과를 들고 chatbot으로 돌아와 → ⑤ 이번엔 검색 결과로 답을 만드니 도구 호출이 없고 → ⑥ END로 끝난다. 이 "LLM ↔ 도구" 왕복이 도구 호출의 핵심이다.

```python
response = graph.invoke({"messages" : [("user", query)]})
print("AI : ", response["messages"][-1].content)   # 마지막 메시지(최종 답변)
```
`messages[-1]`은 쌓인 메시지 중 **맨 마지막**, 즉 최종 답변이다.

**정확한 이해를 위한 보충(추측 포함).** 이 파일은 `llm_with_tools = llm.bind_tools(tools)`로 도구를 묶어 두었지만, 정작 `chatbot` 노드 안에서는 `llm.invoke(...)`로 **도구가 묶이지 않은 원래 llm**을 호출한다. 도구 호출은 "도구가 묶인 LLM"만 만들어 내므로, 이 상태 그대로면 LLM이 도구 호출을 만들지 못해 `tools_condition`이 항상 END로 빠지고 **실제 검색이 일어나지 않을 가능성이 크다.** 도구가 실제로 작동하게 하려면 노드에서 `llm_with_tools`를 호출해야 한다.
```python
def chatbot(state : State):
    return {"messages" : [llm_with_tools.invoke(state["messages"])]}   # llm → llm_with_tools
```
(강사가 다음 단계에서 고치려고 남겨둔 것일 수 있으니, 수업 진행을 확인하고 의도와 맞춰 보면 된다.)

---

## 3. lg06 — 검색 도구 바꾸기·추가하기 (DuckDuckGo)

도구는 Tavily 하나만 쓰는 게 아니다. 다른 검색 도구로 바꾸거나, 여러 개를 함께 붙일 수 있다.

### 3-1. lg06_01 — Tavily 대신 DuckDuckGo

```bash
pip install -U ddgs
```
```python
from langchain_community.tools import DuckDuckGoSearchRun
web_search = DuckDuckGoSearchRun(max_result=5)

result = web_search.invoke({"query" : "삼성전자의 오늘 주가"})
print(result)
exit()   # 여기서 멈춤 — 도구 단독 동작만 확인
```

- **DuckDuckGo 검색은 API 키가 필요 없고 무료**라는 점이 Tavily와 다르다(`ddgs` 패키지만 설치하면 된다). 쓰는 방식(`도구.invoke({"query": ...})`)은 Tavily와 똑같다. **도구만 바꿔 끼우면 나머지 그래프 구조는 그대로** 쓸 수 있다.
- **`exit()`** 는 프로그램을 그 줄에서 멈춘다. 아래 그래프 코드까지 가지 않고, 검색 도구가 잘 동작하는지만 먼저 확인하려고 둔 것이다.

### 3-2. lg06_02 — 도구 두 개를 함께

```python
web_search1 = TavilySearch(max_results=3)       # 도구 1
web_search2 = DuckDuckGoSearchRun(max_result=5) # 도구 2
...
tools = [web_search1, web_search2]   # 리스트에 둘 다
llm_with_tools = llm.bind_tools(tools)
```

- **도구를 리스트에 여러 개 담아 `bind_tools`에 넘기면, LLM이 상황에 맞는 도구를 골라 쓸 수 있다.** `ToolNode(tools)`도 이 목록을 받아 어떤 도구 요청이 와도 실행한다. 도구가 늘어도 그래프 구조(노드·조건부 엣지)는 바뀌지 않는다. (이 파일도 `exit()`로 묶는 부분까지만 확인하고 그래프 실행 전에 멈춰 둔다.)

---

## 4. lg07 — 그래프를 그림으로 저장하기

lg05와 그래프 구성은 같고, **만든 그래프 구조를 이미지(PNG)로 저장**하는 코드가 더해졌다. 어제 본 그래프 그림들이 이렇게 만들어진다.

```python
path = './_save/image/'
png_data = graph.get_graph().draw_mermaid_png()   # 그래프를 PNG 이미지 데이터로

file = open(path + "lg07_ToolNode1.png", "wb")    # "wb" = 이진 쓰기 모드
file.write(png_data)                               # 파일로 저장
file.close()                                       # 파일 닫기
```

- **`graph.get_graph().draw_mermaid_png()`** 는 완성된 그래프의 구조(노드와 엣지)를 **그림 데이터로 그려준다.** 내 그래프가 어떻게 연결됐는지 눈으로 확인할 때 쓴다.
- **파일 저장 3단계**: `open(경로, "wb")`로 파일을 열고(`wb`는 이미지 같은 이진 데이터를 쓰는 모드), `write`로 내용을 쓰고, `close`로 닫는다. (쓰고 나서 닫지 않거나 `write`를 주석 처리하면 빈 파일이 생긴다.)

---

## 5. 한 줄 요약

> **도구 호출(Tool Calling)** 은 LLM이 모르는 실시간·최신 정보를 **외부 도구(웹 검색 등)로 가져와** 답하게 하는 것이다. `bind_tools`로 LLM에 도구 목록을 쥐어주면 LLM이 "도구가 필요한지" 판단해 도구 호출 요청을 만들고, **`ToolNode`** 가 그 도구를 실제로 실행한다. 둘을 잇는 것이 조건부 엣지 **`tools_condition`** 으로, 도구 호출이 있으면 `tools`로, 없으면 `END`로 보낸다. `tools → chatbot`으로 되돌리면 "LLM ↔ 도구" 왕복(순환)이 완성된다. 검색 도구는 Tavily(API 키 필요)·DuckDuckGo(무료) 등으로 바꾸거나 여러 개를 함께 붙일 수 있고, 그래프 구조는 그대로 둔 채 도구만 교체하면 된다. 만든 그래프는 `draw_mermaid_png()`로 그림 저장해 구조를 눈으로 확인할 수 있다. (단, 도구가 실제로 작동하려면 노드에서 도구가 묶인 `llm_with_tools`를 호출해야 한다.)
