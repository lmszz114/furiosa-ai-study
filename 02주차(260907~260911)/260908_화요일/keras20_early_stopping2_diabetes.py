from sklearn.datasets import fetch_california_housing, load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

#1. 데이터
datasets = load_diabetes() # (442, 10) (442,)
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    # train_size=0.5,
    random_state=32,
)

#2. 모델구성
model = Sequential()
model.add(Dense(64,activation='relu', input_dim=10))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 64,
    restore_best_weights = True,
)

hist = model.fit(x_train, y_train, 
                 verbose=1, 
                 epochs=300, 
                 batch_size=4, 
                 validation_split=0.32,
                 callbacks=[es],
                 )

print("===================================")


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss = ', loss)

"""
print("=============== history ====================")
print(hist) # 랩핑 데이터 출력
print("=============== hist.history ====================")
print(hist.history) # 딕셔너리 형태로 출력됨
print("=============== loss ====================")
print(hist.history['loss'])
print("=============== val_loss ====================")
print(hist.history['val_loss'])
"""

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'  

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='red', label='loss') 
plt.plot(hist.history['val_loss'], c='blue', label='val_loss') 
plt.legend(loc='upper right')   
plt.title('diabetes loss') 
plt.xlabel('epoch') 
plt.ylabel('loss')  
plt.grid() 
plt.show()

"""
loss =  2940.730224609375
"""