# keras23_softmax1_onehot_iris 내용 카피

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
print(datasets) 
print(datasets.DESCR)
print(datasets.feature_names) 

x = datasets.data
y = datasets['target']
print(x.shape, y.shape) 
print(y)    # 데이터가 뭔지 모르니 한번 확인
print(np.unique(y, return_counts=True)) # 데이터가 라벨별로 몇개 있는지 확인 / (array([0, 1, 2]), array([50, 50, 50]))

############# 원핫3. sklearn - OnHotEncoder, reshape #############
from sklearn.preprocessing import OneHotEncoder
y = y.reshape(-1, 1)  
print(y.shape)

ohe = OneHotEncoder(sparse_output=False)
y = ohe.fit_transform(y) 
print(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size = 0.7,
    random_state= 4096, 
    shuffle = True,
    stratify = y, 
)

print(x_train.shape, x_test.shape) 
print(y_train.shape, y_test.shape) 


#2. 모델 구성
model = Sequential()
# model.add(Dense(64, input_dim=4, activation='relu'))
model.add(Dense(64, input_shape=(4,), activation='relu'))
# input_dim=4 는 input_shape=(4,) 와 같다
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))
"""
원데이터 -> input_shape
(n,4) -> (4,)
(n,100,3) -> (100,3)
(n,100,100,3) -> (100,100,3)
앞에 n 빼고 전부다

"""

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
print('loss = ', result[0])
print('acc = ', round(result[1], 2))  

y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1)
print(y_predict)

y_test = np.argmax(y_test, axis=1)   
print(y_test)

accuracy_score = accuracy_score(y_test, y_predict)
print('acc_score = ', accuracy_score)
print('걸린시간 = ', round(end_time - start_time, 2), "초")

"""
loss =  0.04997675120830536
acc =  1.0

acc_score =  1.0
걸린시간 =  74.39 초
"""