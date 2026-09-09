from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping


#1. 데이터
datasets = load_iris()
print(datasets) # 딕셔너리 구조로 만들어져있는걸 확인 가능
print(datasets.DESCR)
print(datasets.feature_names)   # 컬럼명 출력

x = datasets.data
y = datasets['target']
print(x.shape, y.shape) # (150, 4) (150,)
print(y)    # 데이터가 뭔지 모르니 한번 확인
print(np.unique(y, return_counts=True)) # 데이터가 라벨별로 몇개 있는지 확인 / (array([0, 1, 2]), array([50, 50, 50]))

"""
y 값 현재 -> [0, 0, 1, 2, 0, ...] 이런식으로 되어있음 / (5, )
각 값의 예시는 0 여자 / 1 남자 / 2 외계인
이 형태를
[1, 0, 0]
[1, 0, 0]
[0, 1, 0]
[0, 0, 1]
[1, 0, 0]
이러한 형태로 바꿔줘야함 / (5, 3)
벡터 형태의 데이터가 행렬로 바뀜
왜 바꾸나?
 -> 여기서 2는 1+1이 아님. 1 은 1+0 이 아님.
 -> 0, 1, 2 모두 같은 가치를 가지고 있음. 그래서 가치(밸류값)가 아닌 위치값으로 판단해줘야함.
"""

############# 원핫1. tensorflow - to_categorical #############
# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)  # to_categorical 는 무조건 0컬럼부터 시작하는게 베이스, 1부터 시작하는 데이터면 0컬럼을 강제로 만들어버림 
# print(y)
# print(y.shape)  # (150, 3)


############# 원핫2. pandas - get_dummies #############
# y = pd.get_dummies(y, dtype=int) # dtype=int 넣고 print(y) 돌리면 TRUE, FALSE로 나오던게 0, 1로 나옴 
# print(y)
# print(y.shape)  # (150, 3)

############# 원핫3. sklearn - OnHotEncoder, reshape #############
from sklearn.preprocessing import OneHotEncoder
# y = y.reshape(150, 1)
y = y.reshape(-1, 1)    # -1 넣으면 처음부터 시작하겠단거
print(y.shape)  # (150, 1)

# ohe = OneHotEncoder()  # 혼동행렬(sparse) 형태로 나옴
ohe = OneHotEncoder(sparse_output=False)
y = ohe.fit_transform(y) # y 데이터 쉐이프가 (150,) 라서 터짐 / 사이킷런 OneHotEncoder 쓰려면 2차원 행렬로 재구성해줘야함 -> reshape
# y 값이 현재 벡터 형태이기 때문에 변환해줘야함
# ValueError: Expected 2D array, got 1D array instead:
# Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.
# reshape 는 fit_transform(y) 하기 전에 해줌
print(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size = 0.7,
    random_state= 4096, 
    shuffle = True,
    stratify = y, 
)

print(x_train.shape, x_test.shape)  # (105, 4) (45, 4)
print(y_train.shape, y_test.shape)  # (105, 3) (45, 3)


#2. 모델 구성
model = Sequential()
model.add(Dense(64, input_dim=4, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax')) # [1, 1, 0 ] 같은 값이 안나오도록 막아줌 (모든 값을 더했을 때 1이 넘지 않게 한다)


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=800,
    restore_best_weights=True
    )

start_time = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=4,
          verbose=1,
          validation_split=0.3,
          callbacks=[es],
          )
end_time = time.time()


#4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('loss = ', result[0]) # result 리스트의 0번째 인덱스 출력
print('acc = ', round(result[1], 2))    # result 리스트의 1번째 인덱스 출력

y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1)   # 확률 3개 중 제일 큰 위치 → 0/1/2
print(y_predict)
# [2 2 0 1 1 0 1 1 1 0 2 0 2 1 0 2 2 1 0 0 2 0 0 1 0 2 1 2 2 1 2 0 0 2 1 1 2 2 1 0 2 0 2 2 0]
y_test = np.argmax(y_test, axis=1)         # 원-핫도 위치로 되돌림 → 0/1/2
print(y_test)
# [1 2 0 1 1 0 1 1 1 0 2 0 2 1 0 2 2 1 0 0 2 0 0 1 0 2 1 2 2 1 2 0 0 2 1 1 2 2 1 0 2 0 1 2 0]

accuracy_score = accuracy_score(y_test, y_predict)
print('acc_score = ', accuracy_score)
print('걸린시간 = ', round(end_time - start_time, 2), "초")

"""
loss =  0.04997675120830536
acc =  1.0

acc_score =  1.0
걸린시간 =  74.39 초
"""