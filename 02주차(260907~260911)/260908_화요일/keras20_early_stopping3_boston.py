from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import boston_housing

#1. 데이터
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
# print(x_train.shape, x_test.shape) # (404, 13) (102, 13)
# print(y_train.shape, y_test.shape) # (404,) (102,)


#2. 모델구성
model = Sequential()
model.add(Dense(64,activation='relu', input_dim=13))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(2,activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',  
    patience = 15,    
    restore_best_weights = True,
)

hist = model.fit(x_train, y_train, 
                 verbose=1, 
                 epochs=1, 
                 batch_size=32, 
                 validation_split=0.2,
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
plt.title('boston loss')
plt.xlabel('epoch')
plt.ylabel('loss') 
plt.grid()
plt.show()

"""
loss =  0.4268846809864044
"""