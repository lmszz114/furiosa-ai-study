from sklearn.datasets import load_digits
import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
datasets = load_digits()
print(datasets)
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets['target']
print(x.shape, y.shape) # (1797, 64) (1797,)
print(y)
print(np.unique(y, return_counts=True)) # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180]))

############# 원핫2. 판다스 pd.get_dummies #############
y = pd.get_dummies(y)
print(y)
print(y.shape)  # (1797, 10)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size = 0.7,
    random_state= 4096, 
    shuffle = True,
    stratify = y, 
)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))

print(x_train.shape, x_test.shape)  # (1257, 64) (540, 64)
print(y_train.shape, y_test.shape)  # (1257, 10) (540, 10)


#2. 모델 구성
model = Sequential()
model.add(Dense(128, input_dim=64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=800,
    restore_best_weights=True
    )

start_time = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=4,
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
# print(y_predict)
y_test = np.argmax(y_test, axis=1)
# print(y_test)

accuracy_score = accuracy_score(y_test, y_predict)
print('acc_score = ', accuracy_score)
print('걸린시간 = ', round(end_time - start_time, 2), "초")

"""
loss =  0.09734956920146942
acc =  0.97
17/17 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step 
acc_score =  0.9703703703703703
걸린시간 =  28.82 초

loss =  0.09012431651353836
acc =  0.97
17/17 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step 
acc_score =  0.9740740740740741
걸린시간 =  28.48 초

별차이 없음
"""