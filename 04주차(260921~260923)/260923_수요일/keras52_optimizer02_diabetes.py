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


#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam
#learning_rate = 0.01
# learning_rate = 0.001     # 0.001 - 디폴트
#learning_rate = 0.0001
#learning_rate = 0.005
learning_rate = 0.05

model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))

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
                 epochs=100, 
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
loss =  3154.514404296875
걸린시간 =  12.06 초
"""

"""
러닝레이트 적용 (미갱신)
learning_rate = 0.01
loss =  4144.93701171875
걸린시간 =  12.8 초

learning_rate = 0.0001
loss =  3793.891357421875
걸린시간 =  13.08 초

learning_rate = 0.005
loss =  3386.514892578125
걸린시간 =  12.91 초

learning_rate = 0.05
loss =  4451.544921875
걸린시간 =  12.7 초
"""