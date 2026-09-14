from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.datasets import boston_housing
import numpy as np

#1. 데이터
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
# print(x_train.shape, x_test.shape) # (404, 13) (102, 13)
# print(y_train.shape, y_test.shape) # (404,) (102,)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
#scaler = MinMaxScaler()
# scaler = MaxAbsScaler()
# scaler = StandardScaler()
scaler = RobustScaler()
# 
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))

#2-1. 순차적 모델
model = Sequential()
model.add(Dense(64,activation='relu', input_dim=13))
model.add(Dropout(0.2))
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(8,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(4,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2,activation='relu'))
model.add(Dense(1))
model.summary()
###########################################################

#2-2 함수형 모델
input1 = Input(shape=(13,))
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
dense6 = Dense(2, activation='relu')(drop5)
output1 = Dense(1)(dense6)
model2 = Model(inputs=input1, outputs=output1)
model2.summary()


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
                 epochs=100, 
                 batch_size=32, 
                 validation_split=0.2,
                 callbacks=[es],
                 )

print("===================================")


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss = ', loss)


"""
RobustScaler
loss =  567.0996704101562
"""