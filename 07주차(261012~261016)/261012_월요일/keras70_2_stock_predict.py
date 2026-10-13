import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Conv1D, Flatten, Concatenate

#1. 데이터 ########################################
USE_COLS  = ['고가', '저가', '종가', '거래량', '금액(백만)', '기관', '외인(수량)']
TIMESTEPS = 10

PATH = 'C:/study/_data/stock/'
F_AMORE = PATH + '아모레261012.csv'
F_HYNIX = PATH + '하이닉스261012.csv'
F_SAM   = PATH + '삼성전자261012.csv' 
WEIGHT_PATH = 'C:/study/_save/keras70/keras70_이민수.weights.h5'

def load_csv(path):
    df = pd.read_csv(path, encoding='cp949', thousands=',', index_col=0)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()

amore = load_csv(F_AMORE)
hynix = load_csv(F_HYNIX)
sam   = load_csv(F_SAM)

am = amore[USE_COLS].add_prefix('am_')
hy = hynix[USE_COLS].add_prefix('hy_')
sy = sam[['시가']].rename(columns={'시가': 'sam_open'})
merged = am.join(hy, how='inner').join(sy, how='inner').dropna() 
print('사용 데이터 마지막 날짜:', merged.index.max().date())   # 10/12 여야 정상

x1 = merged[[c for c in merged.columns if c.startswith('am_')]].values
x2 = merged[[c for c in merged.columns if c.startswith('hy_')]].values
N_FEAT = x1.shape[1]

#1-2. 스케일링 ##
scaler1 = MinMaxScaler()
scaler2 = MinMaxScaler()
scaler1.fit(x1)
scaler2.fit(x2)

# 내일 예측용 : 가장 최근 10일 윈도우
x1_pred = scaler1.transform(x1[-TIMESTEPS:]).reshape(1, TIMESTEPS, N_FEAT)
x2_pred = scaler2.transform(x2[-TIMESTEPS:]).reshape(1, TIMESTEPS, N_FEAT)

#2. 모델 ###################
input1 = Input(shape=(TIMESTEPS, N_FEAT))
lstm1 = LSTM(64, activation='relu')(input1)
dense1 = Dense(32, activation='relu')(lstm1)
output1 = Dense(16, activation='relu')(dense1)

input2 = Input(shape=(TIMESTEPS, N_FEAT))
conv1 = Conv1D(64, 2, activation='relu')(input2)
conv2 = Conv1D(32, 2, activation='relu')(conv1)
flat = Flatten()(conv2)
output2 = Dense(16, activation='relu')(flat)

merge = Concatenate()([output1, output2])
m1 = Dense(32, activation='relu')(merge)
m2 = Dense(16, activation='relu')(m1)
last_output = Dense(1)(m2)

model = Model(inputs=[input1, input2], outputs=last_output)
model.load_weights(WEIGHT_PATH)   # ★ 학습된 weight 불러오기

#3. 예측 #######################################
pred = model.predict([x1_pred, x2_pred])
print('=========================================')
print('내일(2026-10-13) 삼성전자 시가 예측:', int(pred[0][0]), '원')
print('=========================================')