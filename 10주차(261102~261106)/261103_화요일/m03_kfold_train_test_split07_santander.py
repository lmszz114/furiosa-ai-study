import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, train_test_split, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

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

# 분류 문제이므로 y는 0/1 라벨 그대로 사용 (sklearn 분류기는 원핫 불필요)
# ############# 원핫2. pandas - get_dummies #############
# y = pd.get_dummies(y, dtype=int)
# print(y)
# print(y.shape)  # (200000, 2)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,
    random_state=2048,
    stratify=y, 
)

from sklearn.preprocessing import (  # noqa: F401
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)

# scaler = MinMaxScaler()
# scaler = MaxAbsScaler()
# scaler = StandardScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)


#2. 모델
model = DecisionTreeClassifier()

#3~4. (컴파일X), 훈련, 평가
scores = cross_val_score(model, x, y, cv=kfold)

print('loss :', scores, '\n평균 loss :', round(np.mean(scores),2))

y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
print(y_predict)
print(y_test)

acc = accuracy_score(y_test, y_predict)
print("cross_val_predict ACC = ", acc)