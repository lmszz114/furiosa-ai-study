import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Conv1D, Flatten, Concatenate
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터 ########################################
USE_COLS  = ['고가', '저가', '종가', '거래량', '금액(백만)', '기관', '외인(수량)']  # ★시가 제외(=y)
TIMESTEPS = 10

PATH = 'C:/study/_data/stock/'
F_AMORE = PATH + '아모레261012.csv'
F_HYNIX = PATH + '하이닉스261012.csv'
F_SAM   = PATH + '삼성전자261012.csv'
WEIGHT_PATH = 'C:/study/_save/keras70/keras70_이민수.weights.h5'   # ★ 제출용 weight

def load_csv(path):
    # cp949(한글), thousands=','("1,640,000"→1640000), 내림차순→오름차순
    df = pd.read_csv(path, encoding='cp949', thousands=',', index_col=0)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()

amore = load_csv(F_AMORE)
hynix = load_csv(F_HYNIX)
sam   = load_csv(F_SAM)

# 종목마다 거래일이 달라서 날짜 교집합(inner join)으로 맞춘다
am = amore[USE_COLS].add_prefix('am_')
hy = hynix[USE_COLS].add_prefix('hy_')
sy = sam[['시가']].rename(columns={'시가': 'sam_open'})
merged = am.join(hy, how='inner').join(sy, how='inner').dropna()
print('병합 shape:', merged.shape, '| 기간:', merged.index.min().date(), '~', merged.index.max().date())

x1 = merged[[c for c in merged.columns if c.startswith('am_')]].values   # (전체, 7)
x2 = merged[[c for c in merged.columns if c.startswith('hy_')]].values   # (전체, 7)
y  = merged['sam_open'].values                                           # (전체,)
N_FEAT = x1.shape[1]

#1-2. 시계열 윈도우 자르기 ########################
#  10일치[i:i+10]를 보고 → 그 다음날 삼성 시가가 정답
X1, X2, Y = [], [], []
for i in range(len(merged) - TIMESTEPS):
    X1.append(x1[i : i + TIMESTEPS])
    X2.append(x2[i : i + TIMESTEPS])
    Y.append(y[i + TIMESTEPS])
X1, X2, Y = np.array(X1), np.array(X2), np.array(Y)
print('X1:', X1.shape, 'X2:', X2.shape, 'Y:', Y.shape)   # (N,10,7) (N,10,7) (N,)

# 내일(10/13) 예측용 : 가장 최근 10일 윈도우
x1_pred = x1[-TIMESTEPS:].reshape(1, TIMESTEPS, N_FEAT)
x2_pred = x2[-TIMESTEPS:].reshape(1, TIMESTEPS, N_FEAT)

#1-3. 스케일링 (x만) ##############################
#  스케일러는 원본 전체 데이터(x1, x2)로 fit 한다. (predict 코드와 기준을 똑같이 맞추기 위함)
#  적용할 땐 2D만 받으니 (N,10,7)→(N*10,7) 폈다가 되돌린다.
scaler1 = MinMaxScaler()
scaler2 = MinMaxScaler()
scaler1.fit(x1)
scaler2.fit(x2)
X1 = scaler1.transform(X1.reshape(-1, N_FEAT)).reshape(-1, TIMESTEPS, N_FEAT)
X2 = scaler2.transform(X2.reshape(-1, N_FEAT)).reshape(-1, TIMESTEPS, N_FEAT)
x1_pred = scaler1.transform(x1_pred.reshape(-1, N_FEAT)).reshape(1, TIMESTEPS, N_FEAT)
x2_pred = scaler2.transform(x2_pred.reshape(-1, N_FEAT)).reshape(1, TIMESTEPS, N_FEAT)

#1-4. 훈련/테스트 분리 (shuffle=True, keras69 방식) #
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(
    X1, X2, Y, train_size=0.8, random_state=4096, shuffle=True,
)

#2. 모델 (앙상블) ################################
# 2-1. 아모레 가지 → LSTM
input1 = Input(shape=(TIMESTEPS, N_FEAT))
lstm1 = LSTM(64, activation='relu')(input1)
dense1 = Dense(32, activation='relu')(lstm1)
output1 = Dense(16, activation='relu')(dense1)

# 2-2. 하이닉스 가지 → Conv1D
input2 = Input(shape=(TIMESTEPS, N_FEAT))
conv1 = Conv1D(64, 2, activation='relu')(input2)
conv2 = Conv1D(32, 2, activation='relu')(conv1)
flat = Flatten()(conv2)
output2 = Dense(16, activation='relu')(flat)

# 2-3. 합치기 → 삼성 시가 1개 출력
merge = Concatenate()([output1, output2])
m1 = Dense(32, activation='relu')(merge)
m2 = Dense(16, activation='relu')(m1)
last_output = Dense(1)(m2)

model = Model(inputs=[input1, input2], outputs=last_output)

#3. 컴파일, 훈련 #################################
es = EarlyStopping(monitor='val_loss', mode='min', patience=50, restore_best_weights=True)
model.compile(loss='mse', optimizer='adam')
model.fit([x1_train, x2_train], y_train,
          epochs=1000, batch_size=16, validation_split=0.2, callbacks=[es])

model.save_weights(WEIGHT_PATH)   # ★ 가중치 저장
print('weight 저장 완료:', WEIGHT_PATH)

#4. 평가, 예측 ##################################
loss = model.evaluate([x1_test, x2_test], y_test)
y_predict = model.predict([x1_test, x2_test])
r2 = r2_score(y_test, y_predict)
rmse = np.sqrt(mean_squared_error(y_test, y_predict))
print('loss:', loss)
print('R2 :', r2)
print('RMSE:', round(rmse), '원')

# ★ 내일(2026-10-13) 삼성전자 시가 예측
tomorrow = model.predict([x1_pred, x2_pred])
print('=========================================')
print('내일(2026-10-13) 삼성전자 시가 예측:', int(tomorrow[0][0]), '원')
print('=========================================')