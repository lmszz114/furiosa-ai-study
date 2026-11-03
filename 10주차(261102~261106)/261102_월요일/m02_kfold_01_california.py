import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.tree import DecisionTreeRegressor


#1. 데이터
x, y = fetch_california_housing(return_X_y=True)
print(x, y)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)


#2. 모델
model = DecisionTreeRegressor()

#3~4. (컴파일X), 훈련, 평가
scores = cross_val_score(model, x, y, cv=kfold)

print('loss :', scores, '\n평균 loss :', round(np.mean(scores),2))