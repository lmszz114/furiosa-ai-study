from tensorflow.keras.datasets import reuters
import numpy as np
import pandas as pd
import time
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

num_words = 1000
maxlen = 150

(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words = num_words,   # 단어사전의 갯수, 빈도수의 갯수가 높은 단어 순으로 1000개 뽑겠다.
    # maxlen = 1000,       # 최대 길이 제한, 설정한 길이보다 긴 데이터는 제외 시킴.
    test_split = 0.2,
)

print(x_train)
print(x_train.shape, y_train.shape) # (8982,) (8982,)
print(x_test.shape, y_test.shape)   # (2246,) (2246,)
print(y_train)  # [ 3  4  3 ... 25  3 25]
print(np.unique(y_train))
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
#  24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]
# 46개의 라벨이 있음 -> 다중분류 (어떤 카테고리의 뉴스인지 맞추는거)

print(type(x_train))    # <class 'numpy.ndarray'>
print(type(x_train[0]))    # <class 'list'>
# 넘파이 배열로 변환해야함
print(len(x_train[0]), len(x_train[1]))  # 87 56

# 길이가 일정하지 않으니, 길이부터 맞춰야함
# 뉴스 기사의 길이 확인
print("뉴스 기사의 최대 길이: ", max(len(i) for i in x_train))  # 뉴스 기사의 최대 길이:  2376
print("뉴스 기사의 최소 길이: ", min(len(i) for i in x_train))  # 뉴스 기사의 최소 길이:  13
print("뉴스 기사의 평균 길이: ", sum(map(len, x_train)) / len(x_train)) # 뉴스 기사의 평균 길이:  145.5398574927633

# [실습] 평가지표: loss, acc / acc 0.67 이상
# 전처리 (패드 시퀀스)
# y 원핫인코딩

pad_x_train = pad_sequences(x_train,
                         padding='pre',
                         maxlen = maxlen,    
                         truncating = 'post' 
                         )

pad_x_test = pad_sequences(x_test, 
                        padding='pre', 
                        maxlen=maxlen, 
                        truncating='post'
                        )

print(pad_x_train)
print(pad_x_train.shape)    # (8982, 150)
print(pad_x_test.shape)    # (2246, 150)


############# 원핫3. sklearn - OnHotEncoder, reshape #############
ohe = OneHotEncoder(sparse_output=False)
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)

print(y_train.shape, y_test.shape)  # (8982, 46) (2246, 46)

#2. 모델
model = Sequential()
model.add(Embedding(input_dim=num_words, output_dim=100, input_length=maxlen))
model.add(LSTM(200))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(46, activation='softmax'))


#3. 컴파일, 훈련
learning_rate = 0.001
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['acc'])

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='auto',
    patience=20,
    verbose=1,
    factor=0.5
)

start_time = time.time()
es = EarlyStopping(
    monitor='val_acc',
    mode='auto',
    patience=100,
    restore_best_weights=True
    )
model.fit(pad_x_train, y_train, epochs=1000, batch_size=100, 
          verbose=1, 
          validation_split=0.2,
          callbacks=[es,rlr], 
          )
end_time = time.time()


#4. 평가, 예측
print("==================model.evaluate==================")
loss = model.evaluate(pad_x_test, y_test, verbose=1)
print('loss: ', loss[0])
print('acc: ', loss[1])

y_predict = model.predict(pad_x_test)
y_predict = np.argmax(y_predict, axis=1) 
y_test = np.argmax(y_test, axis=1) 

acc_score = accuracy_score(y_test, y_predict) 
print("accuracy_score: ", acc_score)
print("훈련 시간: ", round(end_time - start_time, 2), "초")


"""
loss:  1.64255952835083
acc:  0.7537844777107239
71/71 [==============================] - 1s 4ms/step
accuracy_score:  0.7537845057880677
훈련 시간:  144.09 초
"""