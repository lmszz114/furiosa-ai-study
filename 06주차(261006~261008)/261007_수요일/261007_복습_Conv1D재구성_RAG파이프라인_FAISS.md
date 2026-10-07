# 2026.10.07 복습 — 차원 총정리(Conv1D 포함) · Conv1D 재구성 · RAG 파이프라인(검색+생성)·Gradio·FAISS

> 오늘 흐름: 모델별 차원 재정리(Conv1D 추가) → 여러 데이터를 **Conv1D**로 재구성 → RAG를 **"검색만"에서 "검색+LLM 생성"으로** 완성(수동 연결 → 체인 자동화) → **Gradio 웹 챗봇** → **FAISS**(또 다른 VectorDB)로 같은 흐름 반복. 전체 주제: **"Conv1D로 시퀀스 처리 + RAG 챗봇 완성하기"**.

---

## 1. 이론 — 모델별 차원 총정리 (Conv1D 추가)

| 구분 | Dense | Conv2D | Conv1D | RNN | Embedding |
|---|---|---|---|---|---|
| **입력 차원** | 2차 이상 | 4 | 3 | 3 | 2 |
| **출력 차원** | 2차 이상 | 4 | 3 | 2 | 3 |
| **input_shape 숫자 개수** | 1 | 3 | 2 | 2 | (input_dim·output_dim만) |

- **Dense**: 보통 2D지만 더 높은 차원도 받는다(마지막 축 기준 계산). input_shape는 피처 1개.
- **Conv2D**: 4D 입력·출력(이미지). input_shape 3개 `(H,W,C)`.
- **Conv1D**: **3D 입력·출력**(시퀀스). input_shape 2개 `(타임스텝, 피처)` — **RNN과 동일**. 그래서 RNN↔Conv1D 교체가 쉽다.
- **RNN**: 3D 입력 → 2D 출력(마지막 타임스텝). input_shape 2개.
- **Embedding**: 2D 입력 → 3D 출력. input_shape는 길이지만, 실제로 **`input_dim`(단어사전 크기)·`output_dim`(벡터 차원)** 만 신경 쓰면 된다.
- **핵심**: Conv1D와 RNN은 입력이 똑같이 3D `(N, 타임스텝, 피처)`라, 모델 층만 바꿔 서로 재구성할 수 있다.

---

## 2. Conv1D 재구성 실습 (keras68_01~07)

RNN/CNN/DNN 실습 데이터들을 **Conv1D로** 바꿔 돌리는 연습. 핵심은 **데이터를 3D로 reshape**하고 `input_shape`를 맞추는 것.

### 표 데이터 (diabetes·ddarung·cancer·wine·digits)
```python
# 스케일링 다음에 3D로 (피처를 타임스텝으로)
x_train = x_train.reshape(-1, 피처수, 1)   # 예: wine (N,13)→(N,13,1)
model.add(Conv1D(filters=128, kernel_size=2, input_shape=(피처수, 1)))
model.add(Conv1D(128, 2))
model.add(GlobalAveragePooling1D())        # 또는 Flatten → 2D로
model.add(Dense(...))
model.add(Dense(출력))
```
- **`input_shape=(10)`은 에러**: `(10)`은 정수(괄호는 계산용), 튜플은 **쉼표** 필요 → `(10, 1)`. 게다가 Conv1D는 3D라 피처 1을 붙여 `(피처수, 1)`.
- **`GlobalAveragePooling1D`/`Flatten`**: Conv1D 출력(3D)을 2D로 바꿔 Dense에 연결.

### 이미지 (fashion·cifar100)
```python
x_train = x_train.reshape(-1, 28*28, 1)    # (N,28,28)→(N,784,1)  (cifar는 (N,1024,3))
model.add(Conv1D(filters=128, kernel_size=2, input_shape=(28*28, 1)))
```
- **주의**: Conv1D로 바꾸면 **Conv2D·그 뒤 Flatten은 전부 삭제**해야 한다. 둘은 차원이 달라 섞을 수 없음(GAP1D가 2D로 만들면 Conv2D가 4D를 못 받아 에러).

### 결과 — 데이터 종류가 성능을 가른다 (실측, 정직하게)
| 데이터 | 종류 | Conv1D 결과 |
|---|---|---|
| wine (피처13) | 표 | acc 약 0.98 |
| cancer (피처30) | 표 | 높게 나옴(표 데이터 적합) |
| **fashion (28×28)** | 이미지 | **acc 약 0.10 (사실상 랜덤, 실패)** |
| cifar100 (32×32×3) | 이미지 | 낮음(기록 불명확 — *미기록*) |
- **표 데이터는 Conv1D가 잘 맞지만, 이미지를 1D 시퀀스로 펴면 공간 구조가 깨져 실패**(fashion 0.10). LSTM 때 mnist가 0.11로 실패한 것과 같은 교훈 — "이미지=Conv2D". Conv1D는 시계열·표에 적합.

---

## 3. RAG 완성 — "검색"에서 "검색 + LLM 생성"으로

지난 시간엔 VectorDB에서 **유사 문서 검색**까지 했다. 오늘은 거기에 **LLM을 붙여 답변 생성**까지 = RAG 완성.

### rag12_Chroma04_load — 저장된 DB 불러와 검색 (복습)
```python
vector_store = Chroma(embedding_function=embeddings, persist_directory=DB_PATH, collection_name="croma12")
retriever = vector_store.as_retriever(search_kwargs={"k":2})
aaa = retriever.invoke(query)   # 관련 문서 k개 검색
```

### rag13_Chroma_pipeline1 — 모델 연결 (수동 RAG)
검색한 문서를 **직접 프롬프트에 붙여** LLM에 넣는다.
```python
from langchain_openai import ChatOpenAI
model = ChatOpenAI(model='gpt-5-nano', temperature=0, max_tokens=1000, api_key=api_key, base_url=base_url)

# 그냥 물으면 — 모델이 아는 대로만 답 (근거 없음)
response = model.invoke("삼성전자의 창업자는 누구인가요?")

# 검색 결과를 근거로 붙여 물으면 — RAG 수동 버전
query_with_context = f"""
    {aaa[0].page_content}
    위 내용에 근거하여 다음 질문에 답변하세요.
    {query}
"""
response = model.invoke(query_with_context)   # 검색 문서를 근거로 답변
```
- **`ChatOpenAI`**: 답변 생성용 LLM(임베딩 모델과 별개). `temperature=0`(있는 그대로)~`1`(창의적), `max_tokens`=답 길이 상한.
- **RAG의 핵심 원리**: "검색한 문서를 프롬프트에 넣어 그 근거로 답하게 한다." 이걸 손으로 연결한 버전.

### rag14_Chroma_pipeline2 — RAG 체인 자동화
수동으로 붙이던 걸 **LangChain 체인**으로 자동화.
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

prompt = ChatPromptTemplate.from_template("""
다음 컨텍스트를 바탕으로 질문에 답변해 주세요.
컨텍스트 관련 정보가 없다면 "주어진 정보로는 답변할 수 없습니다"라고 말해주세요.
컨텍스트 : {context}
질문 : {input}
답변 :
""")

docu_chain = create_stuff_documents_chain(model, prompt)   # 프롬프트+모델 (검색문서를 context에 채움)
rag_chain  = create_retrieval_chain(retriever, docu_chain) # 검색 + 생성 = RAG 완성

response = rag_chain.invoke({"input": query})
print(response.keys())        # dict_keys(['input', 'context', 'answer'])
print(response['answer'])     # 최종 답변
```
- **`create_stuff_documents_chain`**: 검색된 문서를 프롬프트의 `{context}`에 **밀어넣어(stuff)** 모델에 전달.
- **`create_retrieval_chain(retriever, docu_chain)`**: **검색(retriever) → 생성(docu_chain)** 을 하나로 묶은 **RAG 체인**. `invoke({"input": 질문})` 하면 알아서 검색+답변.
- 결과는 딕셔너리: **`input`(질문)·`context`(검색된 근거)·`answer`(최종 답변)**.
- 프롬프트에 **"정보 없으면 '답변할 수 없습니다'"** 를 넣어 환각(지어내기)을 억제 — RAG의 목적과 맞는 올바른 프롬프트.

---

## 4. Gradio 웹 챗봇 (rag15·rag16)

RAG 체인을 **웹 UI 챗봇**으로 띄운다.
```bash
activate langchain
pip install gradio      # 설치 필요
```
```python
import gradio as gr

def answer_invoke(message, history):      # message=입력, history=대화기록
    response = rag_chain.invoke({"input": message})
    return response['answer']

demo = gr.ChatInterface(fn=answer_invoke, title='Test Chat Bot')
demo.launch()              # 로컬 실행 / launch(share=True)면 공유 링크 생성
```
- **`gr.ChatInterface(fn=함수)`**: 함수 하나만 넘기면 채팅 UI가 자동 생성. 함수는 `(message, history)`를 받아 답 문자열을 반환.
- **`demo.launch(share=True)`**: 외부 공유용 임시 링크 생성(로컬만이면 `launch()`).

### rag16_Chatbot — 커스텀 (프롬프트가 답을 좌우한다)
- 모델을 바꾸고(`gemini-3.8-flash` 등 라우터 모델), 프롬프트를 자유 커스텀한 버전.
- **주의(안티패턴)**: 이 파일의 프롬프트는 **"컨텍스트에 없는 내용은 거짓말로 지어내라, 지어냈다는 말은 하지 마라"** 로 돼 있다. 이는 **RAG의 목적(환각 방지)과 정반대**로, "프롬프트가 답변을 완전히 좌우한다"를 체감시키려는 실습용 예시다. **실무에선 절대 쓰면 안 되는 패턴**(의도적 환각 유도) — 개념 확인용으로만 보면 된다.

---

## 5. FAISS — 또 다른 VectorDB (rag17·rag18)

Chroma 대신 **FAISS**(Facebook 제작, 빠른 유사도 검색 라이브러리)로 같은 흐름을 구현.
```bash
activate langchain
pip install faiss-cpu      # 윈도우는 faiss-gpu 사용 불가 → cpu 버전
```

### 저장 (rag17_FAISS_1_save)
```python
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

# (불러오기·청킹·임베딩은 Chroma 때와 동일)
faiss_index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))  # L2거리 인덱스, 차원=1536
print(faiss_index.d)    # 1536

db = FAISS.from_documents(documents=split_doc1 + split_doc2, embedding=embeddings)
db.save_local(folder_path=DB_PATH, index_name='faiss_index17')   # 로컬 파일로 저장
```
- **`faiss.IndexFlatL2(차원)`**: **L2(유클리드) 거리**로 벡터 유사도를 재는 인덱스. 차원은 임베딩 차원(1536)과 같아야 함.
- **`FAISS.from_documents`** → **`save_local(folder_path, index_name)`**: 파일로 저장(Chroma의 persist_directory와 유사한 역할).

### 불러오기 (rag17_FAISS_2_load)
```python
db = FAISS.load_local(
    folder_path=DB_PATH, index_name="faiss_index17",
    embeddings=embeddings,
    allow_dangerous_deserialization=True,   # pickle 로드 허용(신뢰하는 파일만)
)
aaa = db.similarity_search("삼성전자 창업주에 대해 알려줘", k=2)
```
- **`allow_dangerous_deserialization=True`**: FAISS는 pickle로 저장돼 불러올 때 코드 실행 위험이 있어 명시적 허용이 필요. **내가 만든 신뢰할 수 있는 파일일 때만** 쓴다.

### rag18_FAISS_gradio — FAISS + Gradio 챗봇 (실습)
- FAISS를 `as_retriever`로 바꿔 Chroma 때와 **똑같은 RAG 체인 + Gradio** 구성. VectorDB만 FAISS로 교체했을 뿐 흐름은 동일.

### Chroma vs FAISS
| | Chroma | FAISS |
|---|---|---|
| 제공 | langchain-chroma | Facebook(faiss) + langchain-community |
| 저장 | `persist_directory` + `collection_name` | `save_local(folder_path, index_name)` |
| 불러오기 | `Chroma(embedding_function=...)` | `FAISS.load_local(..., allow_dangerous_deserialization=True)` |
| 공통 | 둘 다 임베딩 벡터를 저장하고 유사도 검색. **retriever·RAG 체인·Gradio는 동일하게 연결** |

---

## 6. 핵심 플로우 (오늘 완성형)

```
[문서] → 청킹 → 임베딩 → VectorDB 저장(Chroma/FAISS)
                                   ↓ 불러오기
질문 → retriever로 유사 문서 검색 → 프롬프트 {context}에 삽입 → LLM 생성(answer)
                                   ↑ create_retrieval_chain으로 자동화
                                   ↓
                              Gradio 웹 챗봇으로 노출
```
- 지난주 "검색까지"에서, 오늘 **"검색 결과를 근거로 LLM이 답변 생성(RAG) → 웹 챗봇"** 까지 완성.

---

## 7. 한눈에 보기 (파일별 recap)

| 파일 | 주제 | 핵심 |
|---|---|---|
| keras68_Conv1D_01~07 | Conv1D 재구성 | 3D reshape·input_shape=(피처,1). 표는 잘 맞고 이미지는 실패(fashion 0.10) |
| rag12_Chroma04_load | 검색 복습 | Chroma 불러와 retriever로 검색 |
| rag13_pipeline1 | 수동 RAG | 검색 문서를 f-string으로 프롬프트에 붙여 `model.invoke` |
| rag14_pipeline2 | RAG 체인 | `create_stuff_documents_chain`+`create_retrieval_chain`, answer/context/input |
| rag15_Chroma_gradio | 웹 챗봇 | `gr.ChatInterface`+`launch()`, pip install gradio |
| rag16_Chatbot | 커스텀 | 모델·프롬프트 교체. "지어내라" 프롬프트는 안티패턴(개념 확인용) |
| rag17_FAISS_1/2 | FAISS | `IndexFlatL2`, `save_local`/`load_local`, faiss-cpu |
| rag18_FAISS_gradio | FAISS 챗봇 | FAISS+RAG 체인+Gradio(VectorDB만 교체, 흐름 동일) |

---

## 8. 오늘 한 줄 요약

> **Conv1D**는 RNN과 같은 3D `(N,타임스텝,피처)` 입력이라, 데이터를 `reshape(-1,피처,1)`하고 `input_shape=(피처,1)`로 맞추면 재구성된다(`(10)`은 튜플 아님 → `(10,1)`). **표 데이터엔 잘 맞지만(wine 0.98) 이미지를 1D로 펴면 실패**(fashion 0.10) — "이미지=Conv2D".
> **RAG 완성**: 검색한 문서를 프롬프트 `{context}`에 넣어 LLM이 그 근거로 답하게 한다. 수동(f-string+`model.invoke`) → **체인 자동화(`create_stuff_documents_chain`+`create_retrieval_chain`)** → 결과는 `input·context·answer`. **Gradio**(`gr.ChatInterface`+`launch`)로 웹 챗봇화한다.
> **VectorDB는 Chroma·FAISS 둘 다 가능**(FAISS는 `IndexFlatL2`·`save_local`/`load_local`, 윈도우는 faiss-cpu). VectorDB만 바뀔 뿐 **retriever→RAG 체인→Gradio 흐름은 동일**하다. 프롬프트가 답변을 좌우하므로(환각 억제 문구 vs 지어내기 유도) 프롬프트 설계가 중요하다.
