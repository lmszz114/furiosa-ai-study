# 69_2 카피

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#1. 데이터
x1_datasets = np.array([range(100), range(301, 401)]).T
                        # 삼성 종가     # 하이닉스 종가
x2_datasets = np.array([range(101,201),     # 원유가
                        range(411,511),     # 환율
                        range(150,250)]).T  # 금시세
x3_datasets = np.array([range(100), range(301, 401),
                        range(77,177), range(33,133)]).T

print(x1_datasets.shape, x2_datasets.shape, x3_datasets.shape) # (100, 2) (100, 3) (100, 4)

y1 = np.array(range(3001, 3101))    # 화성 온도
y2 = np.array(range(13001, 13101))  # 비트코인 가격
print(y1.shape, y2.shape)  # (100,) (100,)

x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    x1_datasets, x2_datasets, x3_datasets, y1, y2,
    train_size = 0.7,
    random_state= 4096, 
    shuffle = True,
    # stratify = y, 
)


# 2-1. 모델1
input1 = Input(shape=(2,))
dense1 = Dense(10, activation='relu', name='han1')(input1)
dense2 = Dense(20, activation='relu', name='han2')(dense1)
dense3 = Dense(30, activation='relu', name='han3')(dense2)
output1 = Dense(5, activation='relu', name='han4')(dense3)
model1 = Model(inputs=input1, outputs=output1)

# 2-2. 모델2
input21 = Input(shape=(3,))
dense21 = Dense(50, name='han21')(input21)
dense22 = Dense(40, name='han22')(dense21)
dense23 = Dense(30, name='han23')(dense22)
dense24 = Dense(20, name='han24')(dense23)
output21 = Dense(3,name='han25')(dense24)
model2 = Model(inputs=input21, outputs=output21)

# 2-3. 모델3
input31 = Input(shape=(4,))
dense31 = Dense(50, name='han31')(input31)
dense32 = Dense(40, name='han32')(dense31)
dense33 = Dense(30, name='han33')(dense32)
dense34 = Dense(20, name='han34')(dense33)
output31 = Dense(3,name='han35')(dense34)
model3 = Model(inputs=input31, outputs=output31)

# 2-4. 모델 합치기
from tensorflow.keras.layers import concatenate, Concatenate

# merge1 = concatenate([output1, output21], name='mg1')
merge1 = Concatenate(name="mg1")([output1, output21, output31])
merge2 = Dense(10, name="mg2")(merge1)
merge3 = Dense(5, name="mg3")(merge2)

# 2-5. 분기1
last_dense1 = Dense(10, name="ld1")(merge3)
last_dense2 = Dense(10, name="ld2")(last_dense1)
last_output1 = Dense(1, name="last1")(last_dense2)

# 2-6. 분기2
last_output2 = Dense(1, name="last2")(merge3)

model = Model(inputs=[input1, input21, input31], 
              outputs=[last_output1, last_output2])


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit([x1_train, x2_train, x3_train], [y1_train, y2_train], epochs=150, batch_size=8)

#4. 평가, 예측
loss = model.evaluate([x1_test, x2_test, x3_test], [y1_test, y2_test])
print('loss: ', loss)

x1_pred = np.array([range(100), range(301, 401)]).T
x2_pred = np.array([range(101,201),   
                        range(411,511),     
                        range(150,250)]).T 
x3_pred = np.array([range(100), range(301, 401),
                        range(77,177), range(33,133)]).T

y1_predict, y2_predict = model.predict([x1_pred, x2_pred, x3_pred])   # 둘로 언패킹

print("=========================================")
print('화성 온도 예측값:', y1_predict[-1])
print("=========================================")
print('비트코인 예측값:', y2_predict[-1])

"""
=========================================
화성 온도 예측값: [3096.5059]
=========================================
비트코인 예측값: [13102.504]
"""