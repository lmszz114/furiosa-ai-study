# import ssl
# ssl._create_default_https_context = ssl._create_default_https_context

from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import time

path = './_save/keras30/'


#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    # train_size=0.8,
    random_state=2048,
)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))

x_train = x_train.reshape(-1, 8, 1, 1)
x_test = x_test.reshape(-1, 8, 1, 1)
print(x_train.shape, x_test.shape)  #(15480, 8, 1, 1) (5160, 8, 1, 1)
print(y_train.shape, y_test.shape)

#2-1 순차적 모델
model = Sequential()
model.add(Conv2D(64, (2,1), input_shape=(8, 1, 1), padding='same'))  
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
model.add(Dense(20, activation='relu'))  
model.summary()

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(monitor='val_loss', mode='min', patience=30, restore_best_weights=True, verbose=1,)

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', save_best_only=True, filepath=path + 'keras30_mcp1.keras', verbose=1,)

model.compile(loss='mse', optimizer='adam')
start_time = time.time()
hist = model.fit(x_train, y_train, verbose=1,
                 epochs=200, batch_size=32, validation_split=0.2,
                 callbacks=[es, mcp])
end_time = time.time()


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss = ', loss)
y_predict = model.predict(x_test)

print('loss = ', loss)
print('걸린시간 = ', round(end_time - start_time, 2), "초")


"""
DNN -> CNN 모델로 재구성해서 테스트
loss =  0.4854887127876282
걸린시간 =  41.05 초
"""