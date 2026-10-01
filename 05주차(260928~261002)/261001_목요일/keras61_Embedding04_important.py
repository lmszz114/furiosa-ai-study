import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, SimpleRNN
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

x = token.texts_to_sequences(docs)
print(x)

############## 패딩 ##############
from tensorflow.keras.preprocessing.sequence import pad_sequences
padded_x = pad_sequences(x,
                         padding='pre',
                         maxlen = 5,    
                         truncating = 'post',
                         )

print(padded_x)
print(padded_x.shape)   # (15, 5)


#2. 모델
from tensorflow.keras.layers import Embedding
model = Sequential()
####################### 임베딩 1 #######################
model.add(Embedding(input_dim=30, output_dim=10, input_length=5))
#                   단어사전의 갯수,     차원,
model.add(SimpleRNN(10))
model.add(Dense(1))
model.summary()
# embedding (Embedding)       (None, 5, 10)             300       
# simple_rnn (SimpleRNN)      (None, 10)                210  

####################### 임베딩 2 #######################
model.add(Embedding(input_dim=30, output_dim=10,))  # input_length 없어도 임베딩에서는 알아서 맞춰줌
model.add(SimpleRNN(10))
model.add(Dense(1))

####################### 임베딩 3 #######################
model.add(Embedding(30, 10))  # 파라미터명 생략 (맨앞이 아웃풋이 아니라 인풋임. 헷갈리지 않게 주의)
# model.add(Embedding(30, 10, 5))  # input_length=5 넣을 의도였으나 안됨 (파라미터 세번째 옵션이 input_length가 아님 )
model.add(SimpleRNN(10))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(padded_x, labels, epochs=3)