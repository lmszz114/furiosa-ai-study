# 261113 복습 — NPU 스케줄링(Overlap/Eager) · Softmax · 경사하강법 · 학습률 · TensorBoard

> 세 묶음이다. ① NPU 서빙의 **스케줄링 방식(Overlap vs Eager)** 비교. ② 활성화 함수의 마지막 **Softmax**, 그리고 신경망이 **어떻게 학습하는지**의 핵심인 **경사하강법·학습률**. ③ 학습 과정을 시각화하는 **TensorBoard**.

## 목차
1. f30 — Overlap vs Eager Scheduling (NPU)
2. keras78_09 — Softmax (활성화 함수 마무리)
3. keras78_10 — 경사하강법(Gradient Descent)
4. keras79 — 학습률(Learning Rate)
5. keras80 — TensorBoard

---

## 1. f30 — Overlap vs Eager Scheduling (NPU)

NPU 서빙 엔진(furiosa-llm)이 요청을 처리하는 **스케줄링 방식 두 가지**를 비교한다.

- **Continuous Batching**(복습): 여러 사용자의 요청을 동적으로 묶어 NPU에서 처리하는 기술.
- **Overlap Scheduling**: NPU가 **현재 배치를 처리하는 동안**, CPU가 **다음 배치를 미리 준비**하도록 겹쳐(overlap) 실행하는 최적화. 서버를 띄울 때 `--enable-overlap-scheduling` 옵션으로 켠다.
- **Eager Scheduling**: 겹치지 않고 **순차 처리**하는 기본(default) 방식.

```bash
# 기본(eager)
furiosa-llm serve furiosa-ai/Qwen3-8B-FP8 --devices npu:5 --host 127.0.0.1 --port 8000
# overlap 켜기
furiosa-llm serve ... --enable-overlap-scheduling
```

```python
# f30_1 — 순차 요청으로 두 모드 비교 (RPS/TPS)
for prompt in prompts:            # 8개 순차
    response = llm.invoke(prompt)
    total_tokens += (response.usage_metadata or {}).get("output_tokens", 0)
# f30_2 — 동시 요청 수(1·2·4·8)별로 두 모드 비교
```

- **실측 결과**: 두 모드의 차이는 **아주 작았다.**
  - 순차(f30_1): eager TPS 90.86 vs overlap 91.73 (거의 동일)
  - 동시 8요청(f30_2): eager 2.62초/611 TPS vs overlap 2.7초/594 TPS (이 경우 eager가 살짝 빠름)
- **해석**: Overlap은 "NPU 연산과 CPU 준비 작업을 겹쳐" 이론상 빨라질 수 있지만, 이 실험 조건에선 차이가 미미했다. 어떤 방식이 유리한지는 **워크로드(요청 패턴·동시성)에 따라 달라진다.** (overlap이 늘 더 빠른 건 아니다.)

---

## 2. keras78_09 — Softmax (활성화 함수 마무리)

어제 다룬 활성화 함수의 마지막. **Softmax는 여러 출력값을 "합이 1인 확률 분포"로 바꾸는** 함수다. **다중 분류(multi-class)의 출력층**에 쓴다.

```python
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))    # 각 값을 지수로 → 전체 합으로 나눔

x = np.arange(1, 5)        # [1, 2, 3, 4]
y = softmax(x)             # 합이 1인 확률들
plt.pie(y, labels=y, shadow=True)   # 파이차트로 비율 확인
plt.show()
```

- **작동 방식**: 각 값을 `exp()`(지수)로 키운 뒤, **전체 합으로 나눈다.** 그래서 모든 출력이 0~1이고 **다 더하면 1**이 된다 → "각 클래스일 확률"로 해석.
- **큰 값일수록 더 큰 확률**을 갖고, 차이가 과장돼 "가장 큰 것"이 두드러진다. 파이차트로 그리면 각 값이 전체에서 차지하는 비율이 보인다.
- **용도 비교**: **Sigmoid**는 이진 분류(둘 중 하나, 확률 하나), **Softmax**는 **다중 분류**(여러 클래스의 확률 분포). 그래서 다중분류 모델의 마지막 층은 `Dense(클래스수, activation='softmax')`다.

---

## 3. keras78_10 — 경사하강법(Gradient Descent)

신경망이 **어떻게 학습하는지**의 핵심 원리다. 손실(loss)을 가장 작게 만드는 지점을 찾아가는 방법이 **경사하강법**이다.

### 3-1. gradient1 — 손실 함수 모양 보기

```python
f = lambda x : x**2 - 4*x + 6     # 아래로 볼록한 이차함수
x = np.linspace(-1, 6, 100)
plt.plot(x, f(x), 'k-'); plt.grid()
```
- `f(x) = x² − 4x + 6`은 **아래로 볼록한 U자 곡선**이고, **최소점은 x=2**다(이때 f(x)=2). 이 곡선을 "손실 함수"라고 보면, 우리의 목표는 **이 골짜기 바닥(최소점)을 찾는 것**이다.

### 3-2. gradient2 — 경사를 따라 내려가기

```python
f = lambda x : x**2 - 4*x + 6
gradient = lambda x : 2*x - 4     # f(x)의 기울기(미분)

x = -10.0            # 시작점 (아무 데나)
learning_rate = 0.1
epochs = 50

for i in range(epochs):
    x = x - learning_rate * gradient(x)    # ★ 핵심 공식
    print(x, f(x))
```

- **`gradient(x) = 2x − 4`**: 손실 함수 `f(x)`를 미분한 것 = 그 지점의 **기울기(경사)**. 기울기는 "어느 방향으로 올라가는지"를 알려준다.
- **핵심 공식 `x = x - learning_rate * gradient(x)`**: 기울기의 **반대 방향(내리막)** 으로 조금씩 이동한다. 기울기가 양수면 왼쪽으로, 음수면 오른쪽으로 → 어느 쪽에서 시작하든 **골짜기 바닥(x=2)으로 수렴**한다.
- 시작점 x=−10에서 출발해도 50번 반복하면 x가 **최소점 2에 가까워진다.** 이것이 경사하강법이고, **신경망은 이 방식으로 가중치를 조금씩 고쳐 손실을 줄인다.** (실제 모델의 "학습"이 바로 이 과정이다.)
- (코드의 출력 `format(i+i, ...)`는 epoch 번호가 0·2·4…로 찍히는데, `i+1`의 오타로 보인다. 수렴 과정 자체가 요점이라 번호는 넘어가도 된다.)

---

## 4. keras79 — 학습률(Learning Rate)

**학습률(learning rate)은 "한 번에 얼마나 이동할지"의 보폭**이다. 경사하강법에서 `x = x - lr * gradient`의 `lr`이 그것이다. 이 파일은 미분 없이, **w를 조금 올려보고/내려봐서 손실이 작아지는 쪽을 택하는** 단순화된 방식으로 학습률의 역할을 체험한다.

```python
x = 10; y = 10; w = 0.001; lr = 0.999; epochs = 100

for i in range(epochs):
    hypothesis = x * w                  # 예측값 (y = x*w, 단순 모델)
    loss = (hypothesis - y) ** 2        # 손실 (mse)

    up_loss   = (y - x*(w + lr)) ** 2   # w를 lr만큼 '올려본' 손실
    down_loss = (y - x*(w - lr)) ** 2   # w를 lr만큼 '내려본' 손실

    if up_loss > down_loss:             # 내려본 게 더 작으면
        w = w - lr                      #   w를 내린다
    else:
        w = w + lr                      #   아니면 올린다
```

- **아이디어**: 가중치 `w`를 학습률만큼 **올려본 손실과 내려본 손실을 비교**해, **손실이 더 작아지는 방향**으로 `w`를 움직인다. 경사하강법의 직관(손실 줄어드는 쪽으로 이동)을 미분 없이 흉내 낸 것이다.
- **학습률(보폭)의 중요성**:
  - **너무 크면**: 최소점을 건너뛰며 왔다 갔다 하거나 발산(수렴 못 함).
  - **너무 작으면**: 조금씩만 움직여 학습이 매우 느리다.
  - 적절한 학습률을 고르는 것이 학습의 핵심이다. (이 코드의 `lr=0.999`는 꽤 큰 값이라, 보폭이 클 때 어떻게 움직이는지 관찰용이다.)

---

## 5. keras80 — TensorBoard

**TensorBoard는 학습 과정(손실·정확도 변화 등)을 그래프로 보여주는 시각화 도구**다. 콜백으로 붙이면 학습하며 기록이 쌓이고, 브라우저에서 그래프로 확인한다. (MNIST CNN 모델에 적용)

```python
from tensorflow.keras.callbacks import TensorBoard

tb = TensorBoard(
    log_dir='C:/study/_save/tensorboard/_keras80/',   # 기록을 저장할 폴더
    histogram_freq=0,
    write_graph=True,      # 모델 구조 그래프 저장
    write_images=True,     # 가중치를 이미지로 저장
)

model.fit(x_train, y_train, epochs=200, batch_size=128,
          validation_split=0.2, callbacks=[es, tb])   # ← 콜백에 추가
```

- **`TensorBoard(log_dir=...)`**: 학습 로그를 저장할 폴더를 지정하고, `model.fit`의 `callbacks`에 넣는다. 그러면 에폭마다 loss·acc 등이 그 폴더에 기록된다.
- **보는 법**(터미널): 로그 폴더 경로로 들어가 아래를 실행하면 웹 주소(보통 localhost:6006)가 뜨고, 브라우저에서 학습 곡선을 본다.
```bash
tensorboard --logdir=.
# 또는 전체 경로로
tensorboard --logdir=c:/study/_save/tensorboard/
```
- **왜 쓰나**: `print`로 숫자만 보는 것보다, **loss/acc가 에폭에 따라 어떻게 변하는지 그래프로** 보면 과적합·학습 정체 등을 한눈에 파악할 수 있다.
- (이 모델은 지난 MNIST CNN 코드(스케일링·OneHot·Conv2D·softmax 출력)를 재사용하고, 거기에 TensorBoard 콜백만 더한 것이다.)

---

## 한 줄 요약

> **NPU 스케줄링**: **Overlap**(NPU 연산 중 CPU가 다음 배치 준비, `--enable-overlap-scheduling`)과 **Eager**(순차, 기본)를 비교했는데 이 조건에선 차이가 미미했다(워크로드에 따라 다름). **Softmax**는 여러 출력을 합이 1인 확률 분포로 바꿔 **다중분류 출력층**에 쓴다(Sigmoid의 다중 버전). **경사하강법**은 손실 함수의 기울기(미분) 반대 방향으로 `x = x - lr*gradient`만큼 조금씩 이동해 최소점을 찾는 신경망 학습의 핵심 원리이고, **학습률(lr)** 은 그 보폭으로 너무 크면 발산·너무 작으면 느리다. **TensorBoard**는 `TensorBoard(log_dir=...)` 콜백으로 학습 과정을 기록하고 `tensorboard --logdir=경로`로 loss·acc 곡선을 브라우저에서 확인하는 시각화 도구다.
