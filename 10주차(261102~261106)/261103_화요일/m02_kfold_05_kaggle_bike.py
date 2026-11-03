import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.tree import DecisionTreeRegressor


#1. 데이터
path = "./_data/kaggle_bike/"

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
print(train_csv)    # [10886 rows x 11 columns]
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
print(test_csv) # [6493 rows x 8 columns]
submission = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)
print(submission)   # [6493 rows x 1 columns]

print(train_csv.shape, test_csv.shape, submission.shape) # train (10886, 11) / test (6493, 8) / submission (6493, 1)

print(train_csv.info())
print(test_csv.info())

print(train_csv.describe()) # 묘사

######################### 결측치 확인 #########################
print(train_csv.isna().sum()) # 결측치 수치(isna)를 더하기(sum)
print(train_csv.isnull().sum()) # 결측치 위치 찾아서(isnull)를 더하기(sum)


######################### x, y 분리 #########################
x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
print(x)    # [10886 rows x 8 columns]
y = train_csv['count']
print(y, y.shape)    # (10886,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    # train_size=0.8,
    random_state=42,
)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
#scaler = MaxAbsScaler()
# scaler = MinMaxScaler()
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
model = DecisionTreeRegressor()

#3~4. (컴파일X), 훈련, 평가
scores = cross_val_score(model, x, y, cv=kfold)

print('loss :', scores, '\n평균 loss :', round(np.mean(scores),2))