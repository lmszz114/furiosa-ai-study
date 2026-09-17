from sklearn.datasets import fetch_california_housing, load_diabetes
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
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
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))
print(x_train.shape, y_train.shape) # (331, 10) (331,)

x_train = x_train.reshape(-1, 5, 2, 1)
x_test = x_test.reshape(-1, 5, 2, 1)
print(x_train.shape, x_test.shape) # (331, 5, 2, 1) (111, 5, 2, 1)
print(y_train.shape, y_test.shape) # (331,) (111,)


#2-1. 순차적 모델
model = Sequential()
model.add(Conv2D(64, (2,2), input_shape=(5, 2, 1), padding='same'))  
model.add(Conv2D(filters=32, kernel_size=(2,2), activation='relu', padding='same')) 
model.add(Dropout(0.2))
model.add(MaxPooling2D())
model.add(Conv2D(64, (2,1), activation='relu', padding='same')) 
model.add(Dropout(0.2))
model.add(Conv2D(32, (2,1), activation='relu', padding='same'))  
model.add(Dropout(0.2))
model.add(Flatten())
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
                 epochs=200, 
                 batch_size=4, 
                 validation_split=0.32,
                 callbacks=[es],
                 )
end_time = time.time()

print("===================================")


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss = ', loss)
print('걸린시간 = ', round(end_time - start_time, 2), "초")

"""
DNN -> CNN 모델로 재구성해서 테스트

loss =  3787.980224609375
걸린시간 =  36.38 초
"""