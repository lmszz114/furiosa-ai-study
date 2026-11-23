# furiosa-ai-study
퓨리오사 AI 에이전트 전문교육(2026.08.31 ~ 12.24, 16주) 과정에서 매일 배운 코드와 복습 노트를 기록하는 레포지토리

---

## 정리 방식

- **주차별 폴더 > 날짜별 폴더** 구조로 관리
- 각 날짜 폴더에는 그날 실습한 **`.py` 코드**와, 개념을 풀어 정리한 **복습 자료(`.md`)** 를 함께 정리.
- **왜 그렇게 동작하는지**를 복습 노트에 정리하는 데 중점을 둠

```
furiosa-ai-study/
├── 01주차(260831~260904)/
│   ├── 260831_월요일/
│   │   ├── keras01.py ~ keras06.py
│   │   └── 260831_복습_....md
│   ├── ...
│   └── 1주차_총정리.md          # 한 주 내용을 개념 단위로 재구성한 정리본
├── 02주차(260907~260911)/
│   └── ...
└── .gitignore
```

---

## 진행 현황

| 주차 | 기간 | 학습 내용 |
|---|---|---|
| 1주차 | 08.31 ~ 09.04 | TensorFlow/Keras 기초, 선형회귀 → 딥러닝, shape·MLP, train/test 분리, 평가지표(MSE·RMSE·R²), pandas·실전 데이터, relu |
| 2주차 | 09.07 ~ 09.11 | verbose·validation·EarlyStopping, 이진분류(sigmoid), 다중분류(softmax·원-핫), input_shape·파라미터, 스케일러 4종, 모델 save/load |
| 3주차 | 09.14 ~ 09.18 | 모델 저장·Dropout·함수형 모델, GPU 세팅, CNN(Conv2D·Flatten·파라미터·MNIST/CIFAR), padding·pooling·GAP, CNN↔DNN 변환, 이미지 데이터(ImageDataGenerator·npy) |
| 4주차 | 09.21 ~ 09.23 | 이미지 분류 전체 워크플로우(3단계 파일·train/test 분리·자동 라벨링·predict 해석), 내 사진 수치화(img_to_array), 데이터 증폭(datagen.flow·한 클래스 증폭), 러닝레이트(Adam learning_rate·ReduceLROnPlateau), RNN 입문(시계열·타임스텝·SimpleRNN·LSTM 예고) |
| 5주차 | 09.28 ~ 10.02 | 시계열 자르기(split_x·배열 슬라이싱), RNN 심화(GRU·다층 return_sequences·Bidirectional), 차원 구조 총정리(DNN·CNN·RNN·Embedding), 모델 LSTM 재구성·Reshape 층, 자연어 처리(토큰화·원핫·패딩·Embedding), 텍스트 분류(reuters·imdb·sparse_categorical_crossentropy), RAG 입문(LangChain·.env·LCEL·임베딩) |
| 6주차 | 10.06 ~ 10.08 | 시계열→CNN 재구성(Reshape·Conv2D·MaxPooling·OOM 대응), Conv1D로 RNN 재구성(3D 입력·GAP1D), 모델별 차원 총정리, 앙상블(Concatenate·다중 입력/출력 분기), RAG 전체 파이프라인(Chroma·FAISS VectorDB·검색+생성 체인·Gradio), PDF 로더(PyPDFLoader), 임베딩 교체(OpenAI↔HuggingFace) |
| 7주차 | 10.12 ~ 10.15 | 앙상블 분기(1입력→2출력), 앙상블 실전 주식 시가 예측(LSTM+Conv1D·시계열 전처리·weight 저장/predict 분리), 전이학습(가중치·동결 trainable·VGG16 include_top·cifar 동결 비교), 부분 동결·여러 사전학습 모델 for문 비교, RAG 심화(대화기억 MessagesPlaceholder·유사도 임계값·프롬프트 규칙), 검색 방식 3종(BM25·벡터·하이브리드) |
| 8주차 | 10.19 ~ 10.22 | LangGraph 입문(State·Node·Edge·그래프 구성), 도구 호출(ToolNode·조건부 엣지), 메모리·병렬 실행, 병렬 통합·개선, 트리 모델·특성 중요도 |
| 9주차 | 10.26 ~ 10.30 | 미니 프로젝트 기간 (팀 해커톤 진행 — 정규 수업 없음) |
| 10주차 | 11.02 ~ 11.06 | FuriosaAI NPU 실습 시작(furiosa-llm 서빙·OpenAI 호환 API), 함수 호출(tool calling), 모델 양자화, LangChain 스트리밍, 파이썬 모듈/패키지, 교차검증(K-Fold·StratifiedKFold·all_estimators), 하이퍼파라미터 튜닝(GridSearchCV·RandomizedSearchCV) |
| 11주차 | 11.09 ~ 11.13 | NPU LLM 성능 측정(Latency·TTFC·TPS·RPS·Prefill/Decode), 추론 최적화(연속 배칭·KV/프리픽스 캐시·Overlap 스케줄링·서버 메트릭), HalvingGridSearchCV, 파이썬 map·lambda, 활성화 함수 8종, Softmax·경사하강법·학습률·TensorBoard |
| 12주차 | 11.16 ~ 11.20 | 결측치 처리(보간·SimpleImputer·KNNImputer·IterativeImputer), pandas loc/iloc, Tool Calling(@tool·bind_tools·tool_calls)·파이썬 데코레이터, 배치 정규화(BatchNormalization), 퍼셉트론과 XOR 문제(단층 한계·다층 MLP) |

*(매주 업데이트 예정)*

---

## 주차별 총정리

- [1주차 총정리](./01주차(260831~260904)/1주차_총정리.md)
- [2주차 총정리](./02주차(260907~260911)/2주차_총정리.md)
- [3주차 총정리](./03주차(260914~260918)/3주차_총정리.md)
- [4주차 총정리](./04주차(260921~260923)/4주차_총정리.md)
- [5주차 총정리](./05주차(260928~261002)/5주차_총정리.md)
- [6주차 총정리](./06주차(261006~261008)/6주차_총정리.md)
- [7주차 총정리](./07주차(261012~261016)/7주차_총정리.md)
- [8주차 총정리](./08주차(261019~261023)/8주차_총정리.md)
- [10주차 총정리](./10주차(261102~261106)/10주차_총정리.md)
- [11주차 총정리](./11주차(261109~261113)/11주차_총정리.md)
- [12주차 총정리](./12주차(261116~261120)/12주차_총정리.md)

---

* 데이터 파일(csv 등)은 용량 문제로 저장소에 포함하지 않음
