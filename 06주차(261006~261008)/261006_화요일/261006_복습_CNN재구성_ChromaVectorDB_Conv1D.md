# 2026.10.06 복습 — jena CNN 재구성 · RAG VectorDB(Chroma) · Conv1D

> 오늘 흐름: 시계열(jena)을 **CNN(Conv2D)** 으로 재구성 → **RAG의 핵심 플로우**(불러오기→청킹→임베딩→저장→불러오기→유사도검색)를 Chroma VectorDB로 구현 → RNN을 **Conv1D**로 재구성. 전체 주제: **"시계열을 합성곱으로 다루기 + 문서를 벡터로 저장·검색하기"**.

---

## 1. keras66_jena_CNN — 시계열을 CNN으로 재구성

jena(144 타임스텝 × 13 피처)를 **1채널 이미지**로 보고 Conv2D로 처리한다. 데이터 전처리(split_x·x/y 분리·스케일링)는 LSTM 때와 동일하고, **모델 부분만** 바뀐다.

```python
from tensorflow.keras.layers import Reshape, Conv2D, MaxPooling2D, Flatten

model = Sequential()
model.add(Reshape(target_shape=(144, 13, 1), input_shape=(144, 13)))  # 3D → 4D (이미지화)
model.add(Conv2D(64, (2,2), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(32, (2,2), activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
...
model.add(Dense(1))
```

- **모델 안에서 Reshape**: 데이터를 `(N,144,13)` 그대로 두고, 첫 층 `Reshape`로 `(144,13,1)` 4차원(이미지)으로 바꿔 Conv2D에 넣는다. (144=높이, 13=너비, 1=채널)
- **Reshape의 `-1` 주의**: `target_shape=(-1,144,13,1)`처럼 `-1`을 넣으면 차원이 하나 더 생겨 5D가 돼 에러. `target_shape=(144,13,1)`로 정확히.

### 자주 났던 문제 — OOM(메모리 부족)
- `batch_size=7000`에서 `ResourceExhaustedError`(OOM) 발생. Conv2D는 중간 특성맵이 커서 메모리를 많이 먹는다.
- **대응**: `batch_size`를 256으로 낮춤 + **`MaxPooling2D`** 추가(특성맵을 절반으로 줄여 메모리·연산 감소). CNN은 LSTM보다 batch를 작게 써야 한다.

### 결과 — CNN vs LSTM (실행값, 정직하게)
| 모델 | R² | RMSE | 걸린 시간 |
|---|---|---|---|
| LSTM (이전) | 0.9987 | 0.117 | 약 3347초 |
| **CNN (오늘)** | 0.9818 | 0.446 | **약 623초** |
- CNN이 **R²는 약간 낮지만(0.998→0.982)**, **5배 이상 빠르다**(3347초→623초). "정확도 조금 양보하고 속도 크게 얻는" 트레이드오프를 보여주는 실습.

---

## 2. ★ RAG 핵심 플로우 — Chroma VectorDB (rag11·rag12)

오늘 가장 중요한 핵심. **문서를 벡터로 바꿔 DB에 저장하고, 질문과 비슷한 문서를 검색**하는 전체 흐름이다.

```
① 데이터 불러오기 → ② 자른다(청킹) → ③ 임베딩(벡터화) → ④ 저장(VectorDB)
                                                              ↓
                              ⑥ 벡터 유사도로 검색  ←  ⑤ 불러오기
```

### 사전 준비 — 데이터 다운로드 & pip 설치
```bash
# txt 데이터 (구글 드라이브에서 받아 ./_data/rag_data/ 에 둠)
#   samsung_outlook.txt, nvidia_outlook.txt, AI_for_All.txt

# langchain 가상환경에서 설치
pip install langchain-community    # TextLoader 등 문서 로더
pip install langchain-chroma       # Chroma VectorDB 연동
```

### rag11_Chroma01_save — 저장까지
```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# ① 불러오기
loader1 = TextLoader(path + "samsung_outlook.txt", encoding="utf-8")

# ② 청킹
Text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300, chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""],
)
split_doc1 = loader1.load_and_split(Text_splitter)   # 불러오기+자르기 한 번에
print(len(split_doc1), len(split_doc2))   # 9 9  (청크 개수 = document 개수)

# ③ 임베딩
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key, base_url=base_url)

# ④ VectorDB에 저장
DB_PATH = "./_db/Chroma11/"
db = Chroma.from_documents(
    documents = split_doc1 + split_doc2,
    embedding = embeddings,
    persist_directory = DB_PATH,      # 디스크에 저장할 경로
    collection_name = "croma11",      # DB 안의 컬렉션(테이블) 이름
)
```
- **`TextLoader(..., encoding="utf-8")`**: txt를 불러온다(한글이라 utf-8 지정).
- **청킹**: `chunk_size=300`은 **상한**이고, `separators` 순서(문단→줄→단어→글자)로 **의미 경계를 지키며** 자른다. 그래서 각 청크 길이가 300 딱이 아니라 **300 이하로 들쭉날쭉**(290·282·9 등)하다. `chunk_overlap=100`은 청크끼리 100자 겹쳐 문맥 끊김을 완화.
- **`Chroma.from_documents(...)`**: 청크들을 임베딩해 **VectorDB에 저장**. `persist_directory`에 파일로 남아 다음에 불러올 수 있다.

### rag11_Chroma02_load — 불러오기 & 유사도 검색
```python
# ⑤ 저장된 DB 불러오기 (from_documents 아님! 그냥 Chroma(...))
db = Chroma(
    embedding_function = embeddings,   # 저장할 때와 같은 임베딩
    persist_directory = DB_PATH,
    collection_name = "croma11",
)
print(db.get())   # 저장된 내용 확인

# ⑥ 유사도 검색
aaa = db.similarity_search("삼성전자 사업전망에 대해 알려줘", k=2)  # k=개수(기본 4)
print(aaa)
```
- **저장은 `Chroma.from_documents`, 불러오기는 `Chroma(embedding_function=...)`** 로 구분. 불러올 땐 `persist_directory`·`collection_name`이 저장 때와 **같아야** 그 DB를 연다.
- **`similarity_search(질문, k)`**: 질문을 벡터로 바꿔, DB에서 **벡터가 가장 가까운(의미가 비슷한) 청크 k개**를 반환. `k`는 가져올 문서 수(기본 4).

### rag12_Chroma03_save — 폴더의 여러 파일 한 번에
```python
from glob import glob

txt_files = glob(os.path.join(path, "*.txt"))   # 폴더 내 모든 .txt 경로

data = []
for text_file in txt_files:
    loader = TextLoader(text_file, encoding="utf-8")
    data += loader.load()                        # 파일마다 불러와 리스트에 누적
print(len(data))                                  # 3 (파일 3개)

texts = Text_splitter.split_documents(data)       # 여러 문서를 한 번에 청킹
print("청크수:", len(texts))                       # 59  ← 벡터가 59개 생길 예정

vector_store = Chroma.from_documents(texts, embedding=embeddings,
                                     persist_directory="./_db/Chroma12/", collection_name="croma12")
print(vector_store._collection.count())           # 59 (청크 수와 동일)
```
- **`glob("*.txt")`**: 폴더에서 **여러 파일을 한꺼번에** 가져온다. `for`로 각각 `load()`해 `data`에 모은다.
- **`load_and_split` vs `split_documents`**: 전자는 한 파일 불러오며 자르기, 후자는 **이미 불러온 문서 리스트를 한 번에** 자르기. 여러 파일이면 후자가 편하다.
- **document 구조**: `page_content`(본문 내용) + `metadata`(파일 경로 등). `texts[0].page_content`로 내용, 길이는 `len(...)`.

### Retriever(검색기)
```python
retriever = vector_store.as_retriever(search_kwargs={"k": 2})  # 검색기로 변환
aaa = retriever.invoke(query)     # 질문으로 관련 문서 검색
print(len(aaa))                    # 2
```
- **`as_retriever`**: VectorDB를 **검색기(retriever)** 로 바꾼다. `similarity_search`와 하는 일은 비슷하지만, **체인(LLM과 연결)에 끼우기 좋은 표준 인터페이스**다. `search_kwargs={"k":2}`로 개수 조절, `.invoke(질문)`으로 실행.
- 이 retriever가 RAG의 "R(검색)" 부분 — 검색한 문서를 LLM 프롬프트에 붙여 답을 생성(G)하는 게 RAG 완성형(다음 단계).

---

## 3. keras67 Conv1D — RNN을 1D 합성곱으로 재구성

시계열(3차원 `(N, 타임스텝, 피처)`)을 **Conv1D**로 처리한다. RNN과 입력 차원이 같아서, 모델 층만 바꾸면 된다.

### Conv1D란
- **Conv1D**: 1차원(시간축) 방향으로 **커널(필터)이 슬라이딩**하며 지역 패턴을 뽑는 합성곱. CNN(Conv2D)이 이미지의 2D 격자를 훑는다면, Conv1D는 **시퀀스를 한 방향으로** 훑는다.
- 입력은 RNN과 동일한 **3차원 `(N, 타임스텝, 피처)`**. 출력도 3차원이라 **Flatten 또는 GlobalAveragePooling1D**로 2D로 바꿔 Dense에 연결한다.

```python
# keras67_Conv1D_1
model.add(Conv1D(filters=10, kernel_size=2, input_shape=(3,1)))
model.add(Conv1D(10, 2))
model.add(Flatten())          # 3D → 2D
model.add(Dense(20, activation='relu'))
...
model.add(Dense(1))
```
```python
# keras67_Conv1D_2_scale, _3_jena
model.add(Conv1D(filters=20, kernel_size=2, input_shape=(3,1)))
model.add(Conv1D(20, 2))
model.add(GlobalAveragePooling1D())   # 3D → 2D (각 피처의 평균)
...
```
- **`Conv1D(filters, kernel_size, input_shape)`**: `filters`=뽑을 특성 수, `kernel_size`=한 번에 보는 타임스텝 수(예: 2면 연속 2개씩).
- **Flatten vs GlobalAveragePooling1D**: 둘 다 3D→2D로 만들어 Dense에 넘기는 역할. Flatten은 전부 펼치고, GAP1D는 타임스텝 축을 평균내 압축(파라미터가 적어짐).

### 결과 — Conv1D vs RNN (실행값)
| 실습 | RNN 결과 | Conv1D 결과 |
|---|---|---|
| Conv1D_1 ([8,9,10]→11) | LSTM 10.76 | **11.04** |
| Conv1D_2 ([50,60,70]→80) | LSTM 79.85 | **80.13** |
| Conv1D_3 jena (T 예측) | LSTM R² 0.9987 | (GPU 결과 미기록) |

- 간단한 수열 예측에서 **Conv1D가 RNN과 비슷하거나 살짝 더 정확**하게 나왔다.
- **오늘의 교훈: 시계열 데이터에 Conv1D가 생각보다 잘 맞는 경우가 많다.** RNN은 타임스텝을 순차 처리해 느린 반면, Conv1D는 **병렬 처리가 가능해 더 빠르면서** 지역적 시간 패턴을 잘 잡는다. (jena Conv1D GPU 결과는 파일에 안 적혀 있어 수치 비교는 생략 — *미기록*.)

---

## 4. 한눈에 보기 (파일별 recap)

| 파일 | 주제 | 핵심 |
|---|---|---|
| keras66_jena_CNN | 시계열→CNN | Reshape(144,13,1)→Conv2D→MaxPooling, batch 작게(OOM), R² 0.982·623초 |
| rag11_Chroma01_save | 문서 저장 | 불러오기→청킹→임베딩→`Chroma.from_documents`로 VectorDB 저장 |
| rag11_Chroma02_load | 불러오기·검색 | `Chroma(embedding_function=...)`로 열고 `similarity_search(q, k)` |
| rag12_Chroma03_save | 폴더 다중파일 | `glob("*.txt")`→loop load→`split_documents`→저장, `as_retriever` |
| keras67_Conv1D_1/2/3 | RNN→Conv1D | 3D 입력, Flatten/GAP1D로 2D, 시계열에 Conv1D 잘 맞음 |

---

## 5. 오늘 한 줄 요약

> 시계열(jena)을 **CNN(Conv2D)** 으로 재구성하려면 모델 안에서 `Reshape(144,13,1)`로 이미지화 후 Conv2D를 쌓는다. Conv2D는 메모리를 많이 써 **batch_size를 작게(256)·MaxPooling 추가**해야 OOM을 피한다. 결과는 LSTM보다 R²는 약간 낮지만(0.982) **5배 빠르다**.
> **RAG 핵심 플로우**는 **① 불러오기(TextLoader) → ② 청킹(RecursiveCharacterTextSplitter) → ③ 임베딩(OpenAIEmbeddings) → ④ 저장(`Chroma.from_documents`) → ⑤ 불러오기(`Chroma(...)`) → ⑥ 유사도 검색(`similarity_search`/`as_retriever`)**. 저장은 `from_documents`, 불러오기는 `Chroma(embedding_function=...)`로 구분하고, 여러 파일은 `glob`+`split_documents`로 한 번에 처리한다.
> **Conv1D**는 시계열(3D)을 시간축으로 훑는 합성곱으로, RNN과 입력이 같고 **더 빠르면서 성능도 비슷하거나 더 좋은 경우가 많다**(Flatten/GAP1D로 2D 변환 후 Dense 연결).
