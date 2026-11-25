import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from xgboost import XGBRegressor


#1. 데이터
x, y = fetch_california_housing(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=444,
    # stratify=y,
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


parameters = {'n_estimators' : 2000,
             'learning_rate' : 0.05,
             'max_depth' : 7,
             'gamma' : 1,
             'min_child_weight' : 1,
             'subsample' : 1,
             'colsample_bytree' : 1,
             'colsample_bylevel' : 1,
             'colsample_bynode' : 1,
             'reg_alpha' : 0,
             'reg_lambda' : 1,
             'random_state' : 3377,
            #  'verbose' : 0,
            }

#2. 모델
model = XGBRegressor(
    # **parameters
    )

model.set_params(**parameters,
                 early_stopping_rounds=20
                 )

#3. 훈련
model.fit(x_train, y_train,
          eval_set = [(x_test, y_test)],
          verbose = 1,
          )

#4. 평가, 예측
result = model.score(x_test, y_test)
print('최종점수 : ', result)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print("r2 : ", r2)

path = './_save/pickle/'
# import joblib
import pickle
# joblib.dump(model, path + 'm19_joblib1_save.joblib')
pickle.dump(model, open(path + 'm20_pickle1_save.pickle', 'wb'))