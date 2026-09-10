# https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

# path = 'C:/study/_data/kaggle_santander/'
path = './_data/kaggle_santander/'

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)

print(train_csv.shape)  # (200000, 201)
print(test_csv.shape)  # (200000, 200)
print(submission_csv.shape)  # (200000, 1)

# 결측치 확인
# print(train_csv.info()) # 데이터가 많아서 이걸로 보기엔 현실적으로 불가능
print(train_csv.isna().sum())
print(test_csv.isnull().sum())

# x, y 분리
x = train_csv.drop(['target'], axis=1)
y = train_csv['target']
print(x.shape, y.shape) # (200000, 200) (200000,)

print(np.unique(y, return_counts=True)) # (array([0, 1]), array([179902,  20098])) / 0이 179902개, 1이 20098개

############# 원핫2. pandas - get_dummies #############
y = pd.get_dummies(y, dtype=int)
print(y)
print(y.shape)  # (200000, 2)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,
    random_state=2048,
    stratify=y, 
)


#2. 모델구성
model = Sequential()
model.add(Dense(256, activation='relu', input_dim=200))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(2, activation='softmax'))


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              # metrics=['accuracy'],
              metrics=['acc'],
              )

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',  
    patience = 128,    
    restore_best_weights = True,
)

start_time = time.time()
hist = model.fit(x_train, y_train, 
                 verbose=1, 
                 epochs=30, 
                 batch_size=64, 
                 validation_split=0.25,
                 callbacks=[es],
                 )
end_time = time.time()


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss = ', loss)

print("===========================================")
print('loss = ', loss[0])
print('acc = ', round(loss[1],4)) 
print("===========================================")

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

from sklearn.metrics import accuracy_score
acc_score = accuracy_score(y_test, y_pred)
print("acc_score = ", acc_score)

#y_submit = model.predict(test_csv)
# submission_csv['target'] = y_submit
# submission_csv.to_csv(path + "submit/" + "submit_santander.csv")
print('걸린시간 = ', round(end_time - start_time, 2), "초")

"""
===========================================
loss =  0.24217456579208374
acc =  0.9107
===========================================
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 498us/step 
acc_score =  0.9107166666666666
걸린시간 =  68.16 초
"""