# m01_3 카피

from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

#1. 데이터
x, y = load_iris(return_X_y=True)
print(x.shape, y.shape) # (150, 4) (150,)

random_state = 123

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=random_state,
    stratify=y
)

#2. 모델 구성
model = XGBClassifier(random_state=random_state)

#3. 컴파일, 훈련
model.fit(x_train, y_train)
print("=================", model.__class__.__name__, "=================")

#4. 평가, 예측
score = model.score(x_test, y_test)
print("acc : ", score)
print("F Score : ", model.feature_importances_)
# F Score : 빈도수
gain_score = model.get_booster().get_score(importance_type='gain')
print("Gain Score : ", gain_score)
# Gain Score : 손실함수의 개선 점수. 높으면 개선 많이 된거.

import matplotlib.pyplot as plt
from xgboost import plot_importance

# F-score 그래프
# plot_importance(model, importance_type='weight')
# plt.title("F-Score")
# plt.show()

# Gain-score 그래프
plot_importance(model, importance_type='gain')
plt.title("Gain-Score")
plt.show()
