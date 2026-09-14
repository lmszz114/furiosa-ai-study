from sklearn.datasets import fetch_california_housing, load_diabetes
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
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
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
#scaler = MaxAbsScaler()
# scaler = MinMaxScaler()
#scaler = StandardScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))


#2-1. 순차적 모델
model = Sequential()
model.add(Dense(64,activation='relu', input_dim=10))
model.add(Dropout(0.2))
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(8,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(4,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))
model.summary()
###########################################################

#2-2 함수형 모델
input1 = Input(shape=(10,))
dense1 = Dense(64, activation='relu')(input1)
drop1 = Dropout(0.2)(dense1)
dense2 = Dense(32, activation='relu')(drop1)
drop2 = Dropout(0.2)(dense2)
dense3 = Dense(16, activation='relu')(drop2)
drop3 = Dropout(0.2)(dense3)
dense4 = Dense(8, activation='relu')(drop3)
drop4 = Dropout(0.2)(dense4)
dense5 = Dense(4, activation='relu')(drop4)
drop5 = Dropout(0.2)(dense5)
output1 = Dense(1)(drop5)
model2 = Model(inputs=input1, outputs=output1)
model2.summary()



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
RobustScaler
loss =  3008.349853515625
비슷함
"""

"""
loss =  3122.78564453125
"""