from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D
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
scaler = MinMaxScaler()
# 
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))

x_train = x_train.reshape(-1, 13, 1, 1)
x_test = x_test.reshape(-1, 13, 1, 1)
print(x_train.shape, x_test.shape) #(404, 13, 1, 1) (102, 13, 1, 1)
print(y_train.shape, y_test.shape)  #(404,) (102,)


#2-1. 순차적 모델
model = Sequential()
model.add(Conv2D(64, (2,1), input_shape=(13, 1, 1), padding='same'))  
model.add(Conv2D(32, (2,1), activation='relu')) 
model.add(Dropout(0.2))
model.add(Conv2D(64, (2,1), activation='relu', padding='same')) 
model.add(Dropout(0.2))
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(units=32, activation='relu')) 
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='relu'))  
model.summary()


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',  
    patience = 100,    
    restore_best_weights = True,
)
import time

start_time = time.time()
hist = model.fit(x_train, y_train, 
                 verbose=1, 
                 epochs=700, 
                 batch_size=32, 
                 validation_split=0.2,
                 callbacks=[es],
                 )
end_time = time.time()


print("===================================")


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss = ', loss)
print('걸린시간 = ', round(end_time - start_time, 2), "초")


"""
loss =  24.176877975463867
걸린시간 =  22.65 초
"""