from sklearn.datasets import load_wine
import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
datasets = load_wine()
print(datasets)
print(datasets.DESCR)
print(datasets.feature_names)   # 컬럼명 출력
# ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']

x = datasets.data
y = datasets['target']
print(x.shape, y.shape) # (178, 13) (178,)
print(y)    # 데이터가 뭔지 모르니 한번 확인
print(np.unique(y, return_counts=True)) # (array([0, 1, 2]), array([59, 71, 48])) / 0이 59개, 1이 71개, 2가 48개

############# 원핫2. 판다스 pd.get_dummies #############
y = pd.get_dummies(y)
print(y)
print(y.shape)  # (178, 3)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size = 0.7,
    random_state= 4096, 
    shuffle = True,
    stratify = y, 
)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
#scaler = StandardScaler()
# scaler = MinMaxScaler()
# scaler = MaxAbsScaler()
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))

print(x_train.shape, x_test.shape)  # (124, 13) (54, 13)
print(y_train.shape, y_test.shape)  # (124, 3) (54, 3)
x_train = x_train.reshape(-1, 13, 1, 1)
x_test = x_test.reshape(-1, 13, 1, 1)
print(x_train.shape, x_test.shape)  
print(y_train.shape, y_test.shape)

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
model.add(Dense(3, activation='softmax'))  
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
model.fit(x_train, y_train, epochs=100, batch_size=64,
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
loss =  0.5287420749664307
acc =  0.8
2/2 [==============================] - 0s 2ms/step
acc_score =  0.7962962962962963
걸린시간 =  4.68 초
"""