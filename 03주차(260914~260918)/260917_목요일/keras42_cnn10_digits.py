from sklearn.datasets import load_digits
import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D
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

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
#scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = MaxAbsScaler()
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))

print(x_train.shape, x_test.shape)  # (1257, 64) (540, 64)
print(y_train.shape, y_test.shape)  # (1257, 10) (540, 10)
x_train = x_train.reshape(-1, 32, 2, 1)
x_test = x_test.reshape(-1, 32, 2, 1)

#2-1 순차적 모델
model = Sequential()
model.add(Conv2D(64, (2,2), input_shape=(32, 2, 1), padding='same'))  
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
model.add(Dense(10, activation='softmax'))  
model.summary()


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=800,
    restore_best_weights=True
    )

start_time = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=128,
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
acc_score =  0.8425925925925926
걸린시간 =  37.46 초
"""