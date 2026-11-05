import numpy as np
from sklearn.datasets import load_digits, load_iris, load_diabetes
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
import time
import warnings
warnings.filterwarnings('ignore')


#1. 데이터
# load_diabetes 는 회귀 데이터셋(y가 연속값)이라 stratify / StratifiedKFold / Classifier 사용 불가
x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, train_size=0.8,
    # stratify=y,   # 회귀에서는 사용 불가 (연속값이라 클래스당 1개뿐)
)

n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=333)

parameters = [
    {'n_estimators' : [100, 500], 'max_depth' : [6, 10, 12],
     'learning_rate' : [0.1, 0.01, 0.001]}, # 18
    {'max_depth' : [6,8,10,12], 'learning_rate' : [0.1, 0.01, 0.001]}, # 12
    {'min_child_weight' : [2,3,5,10], 'learning_rate' : [0.1, 0.01, 0.001]} # 12
] # 42

#2. 모델
xgb = XGBRegressor()
model = GridSearchCV(xgb, parameters, cv=kfold,
                    verbose=1,
                    n_jobs=-1,
                    )

#3. 훈련
start = time.perf_counter()
model.fit(x_train, y_train)
end = time.perf_counter()

print("최적의 매개변수 : ", model.best_estimator_)
print("최적의 파라미터 : ", model.best_params_)

#4. 평가, 예측
print('model.score :', model.score(x_test, y_test))   # 회귀는 R2 score
print('best_score : ', model.best_score_)

y_predict = model.predict(x_test)
print('r2_score :', r2_score(y_test, y_predict))

print("걸린시간 : ", round(end - start, 2), "초")