# 60_1 카피

from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np

text1 = "나는 지금 진짜 진짜 매우 매우 맛있는 김밥을 엄청 마구 마구 마구 마구 먹었다."
text2 = "개똥이는 기관사를 좋아힌다. 말똥이는 잘생겼다. 길동이는 마구 마구 더 잘생겼다."

token = Tokenizer() #여기서 Token은 객체(인스턴스) / 인스턴스 생성
token.fit_on_texts([text1, text2])

print(token.word_index)
# {'마구': 1, '진짜': 2, '매우': 3, '잘생겼다': 4, '나는': 5, '지금': 6, '맛있는': 7, '김밥을': 8, 
# '엄청': 9, '먹었다': 10, '개똥이는': 11, '기관사를': 12, '좋아힌다': 13, '말똥이는': 14, '길동이는': 15, '더': 16}

x = token.texts_to_sequences([text1, text2])
print(x)
# [[5, 6, 2, 2, 3, 3, 7, 8, 9, 1, 1, 1, 1, 10], [11, 12, 13, 14, 4, 15, 1, 1, 16, 4]]

# x = x[0] + x[1] # x 리스트 연결
x = np.concatenate(x) # x = x[0] + x[1] 대신 사용
print(x)

############# 원핫3. sklearn - OnHotEncoder, reshape #############
from sklearn.preprocessing import OneHotEncoder
# x = np.array(x)   # 위에서 concatenate 사용했으므로 이미 numpy라 np.array 불필요
x = x.reshape(-1, 1)
print(x.shape)  # (24, 1)
ohe = OneHotEncoder(sparse_output=False)
x = ohe.fit_transform(x)
# print(x)
print(x.shape)  # (24, 16)