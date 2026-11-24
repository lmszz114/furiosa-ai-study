import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeRegressor, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestRegressor
from sklearn.ensemble import VotingRegressor, VotingRegressor, StackingRegressor
from xgboost import XGBRegressor, XGBRegressor
from lightgbm import LGBMRegressor, LGBMRegressor
from catboost import CatBoostRegressor, CatBoostRegressor
# pip install catboost
# pip install lightgbm


#1. 데이터
x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=444,
    # stratify=y,
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델
xgb = XGBRegressor()
lg = LGBMRegressor(verbosity=-1)
cat = CatBoostRegressor(verbose=0)
rf = RandomForestRegressor()

model = StackingRegressor(
    estimators=[('XGB', xgb), ('RF', rf), ('CAT', cat), ('LG', lg)],
    final_estimator=CatBoostRegressor(verbose=0),
    n_jobs=1,
    cv=5,
)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print("스태킹 결과 : ", r2)
