import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional
import time
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd

#1. 데이터
docs = [
    '너무 재미있다', '참 최고에요', '참 잘만든 영화에요',
    '추천하고 싶은 영화입니다', '한 번 더 보고 싶어요', '글쎄',
    '별로에요', '생각보다 지루해요', '연기가 어색해요',
    '재미없어요', '너무 재미없다', '참 재밌네요',
    '개똥이 바보', '말똥이 잘생겼다', '길동이 또 구라친다'
]

labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,0,1,0])
# 긍정은 1, 부정은 0으로 라벨링 해놓은 상태

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
# {'참': 1, '너무': 2, '재미있다': 3, '최고에요': 4, '잘만든': 5, '영화에요': 6, '추천하고': 7, '싶은': 8, '영화입니다': 9, '한': 10, 
# '번': 11, '더': 12, '보고': 13, '싶어요': 14, '글쎄': 15, '별로에요': 16, '생각보다': 17, '지루해요': 18, '연기가': 19, '어색해요': 20, 
# '재미없어요': 21, '재미없다': 22, '재밌네요': 23, '개똥이': 24, '바보': 25, '말똥이': 26, '잘생겼다': 27, '길동이': 28, '또': 29, '구라친다': 30}

x = token.texts_to_sequences(docs)
print(x)
# [[2, 3], [1, 4], [1, 5, 6], [7, 8, 9], [10, 11, 12, 13, 14], [15], [16], [17, 18], [19, 20], [21], [2, 22], [1, 23], [24, 25], 
# [26, 27], [28, 29, 30]]
# 각각 길이가 다름
# 빈자리를 가장 긴 데이터를 기준으로, 짧은 데이터들에 0으로 채워넣음.
# 단, 데이터의 뒤가 아닌 앞에 채워줌 (뒤에 0을 채우면 갈수록 맥락이 흐려짐)
# 패딩에 두가지 옵션 존재 (pre, post)

############## 패딩 ##############
from tensorflow.keras.preprocessing.sequence import pad_sequences
padded_x = pad_sequences(x,
                         padding='pre',  # 뒤에 채우려면 post
                         maxlen = 5,    
                         truncating = 'post' # 자르는 경우 디폴트는 앞이 잘림 / post 쓰면 뒤가 잘림 
                         )

print(padded_x)
print(padded_x.shape)   # (15, 5)
padded_x = padded_x.reshape(15,5,1)
print(padded_x.shape)   # (15, 5, 1)

x_train, x_test, y_train, y_test = train_test_split(
    padded_x, labels,
    train_size=0.8,
    random_state=42,
    stratify=labels,
)

print(x_train.shape, x_test.shape)   # (12, 5) (3, 5)
print(y_train.shape, y_test.shape)   # (12,) (3,)


#2. 모델구성
model = Sequential()
model.add(LSTM(256, input_shape=(5,1)))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              # metrics=['accuracy'],
              metrics=['acc'],
              )

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',  
    patience = 9999,    # 안멈추게 하려고 일부러 9999 줬음
    restore_best_weights = True,
)

start_time = time.time()
hist = model.fit(x_train, y_train, 
                 verbose=1, 
                 epochs=100, 
                 batch_size=1, 
                 validation_split=0.2,
                 callbacks=[es],
                 )
end_time = time.time()


#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc : ', acc)

y_predict = ["참 재밌네요"]

pred_seq = token.texts_to_sequences(y_predict)        # [[24, 27]]
pred_pad = pad_sequences(pred_seq,
                         padding='pre',
                         maxlen=5,
                         truncating='post')           # [[0, 0, 0, 24, 27]]

result = model.predict(pred_pad)                      # 0~1 확률
print('예측 확률 : ', result)
print('판정 : ', '긍정(1)' if result[0][0] > 0.5 else '부정(0)')