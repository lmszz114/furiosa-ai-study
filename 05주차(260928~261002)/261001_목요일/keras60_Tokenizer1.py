from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np

text = "나는 지금 진짜 진짜 매우 매우 맛있는 김밥을 엄청 마구 마구 마구 마구 먹었다."

token = Tokenizer() #여기서 Token은 객체(인스턴스) / 인스턴스 생성
token.fit_on_texts([text])

print(token.word_index)
# {'마구': 1, '진짜': 2, '매우': 3, '나는': 4, '지금': 5, '맛있는': 6, '김밥을': 7, '엄청': 8, '먹었다': 9}
# 어절 단위로 잘림

print(token.word_counts)    # 어절 단위 각 개수
# OrderedDict([('나는', 1), ('지금', 1), ('진짜', 2), ('매우', 2), ('맛있는', 1), ('김밥을', 1), ('엄청', 1), ('마구', 4), ('먹었다', 1)])

# text를 수치화 하기
x = token.texts_to_sequences([text])
print(x)
# [[4, 5, 2, 2, 3, 3, 6, 7, 8, 1, 1, 1, 1, 9]]
x = x[0] # 안쪽 리스트만 꺼내기
print(x)
# [4, 5, 2, 2, 3, 3, 6, 7, 8, 1, 1, 1, 1, 9]


#원핫 인코딩 3가지 만들기

############# 원핫1. tensorflow - to_categorical #############
# from tensorflow.keras.utils import to_categorical
# x = to_categorical(x) 
# print(x)
# print(x.shape)  # (1, 14, 10) # 0컬럼 채워짐


############# 원핫2. pandas - get_dummies #############
# x = pd.get_dummies(x, dtype=int) 
# print(x)
# print(x.shape)  # (14, 9)

############# 원핫3. sklearn - OnHotEncoder, reshape #############
from sklearn.preprocessing import OneHotEncoder
x = np.array(x)
x = x.reshape(-1, 1)
print(x.shape)  # (14, 1)
ohe = OneHotEncoder(sparse_output=False)
x = ohe.fit_transform(x) 
# print(x)
print(x.shape)  # (14, 9)