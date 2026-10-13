# 69_3 카피

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

#1. 데이터
x_datasets = np.array([range(100), range(301, 401)]).T   # (100, 2) 삼성·하이닉스

y1 = np.array(range(3001, 3101))     # 화성 온도   (출력1)
y2 = np.array(range(13001, 13101))   # 비트코인 가격 (출력2) ← 2번째 타깃 추가

x_train, x_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    x_datasets, y1, y2,
    train_size=0.7, random_state=4096, shuffle=True,
)

#2-1. 공통 몸통 (입력 1개)
input1 = Input(shape=(2,))
dense1 = Dense(10, activation='relu', name='han1')(input1)
dense2 = Dense(20, activation='relu', name='han2')(dense1)
dense3 = Dense(30, activation='relu', name='han3')(dense2)
output1 = Dense(5, activation='relu', name='han4')(dense3)   # 여기까지 공통, 아래서 갈라짐

#2-2. 분기1 (y1용)
b1 = Dense(10, activation='relu', name='b1_1')(output1)
b1 = Dense(10, activation='relu', name='b1_2')(b1)
last_output1 = Dense(1, name='last1')(b1)

#2-3. 분기2 (y2용)
b2 = Dense(10, activation='relu', name='b2_1')(output1)
b2 = Dense(10, activation='relu', name='b2_2')(b2)
last_output2 = Dense(1, name='last2')(b2)

#2-4. 모델 정의 — 입력 1개, 출력 2개
model = Model(inputs=input1, outputs=[last_output1, last_output2])
# model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, [y1_train, y2_train], epochs=150, batch_size=8)   # x는 1개, y는 리스트

#4. 평가, 예측
loss = model.evaluate(x_test, [y1_test, y2_test])
print('loss:', loss)   # [전체loss, 출력1 loss, 출력2 loss]

y1_predict, y2_predict = model.predict(x_test)   # 출력 2개라 언패킹
print("=========================================")
print('화성 온도 예측:', y1_predict[-1])
print('비트코인 예측:', y2_predict[-1])
print("=========================================")

"""
화성 온도 예측: [3086.0176]
비트코인 예측: [13192.517]
"""