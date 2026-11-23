import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x_data = [[0,0], [0,1], [1,0], [1,1]]
y_data = [0, 1, 1, 0]

#2. 모델
# model = Perceptron()
# model = LinearSVC()
model = Sequential()
model.add(Dense(1, input_dim=2, activation='sigmoid'))

#3. 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_data, y_data, epochs=1000)

#4. 평가, 예측
y_perdict = model.predict(x_data)

result = model.evaluate(x_data, y_data)
print("loss, acc : ", result)

acc = accuracy_score(y_data, np.round(y_perdict))
print("accuracy_score : ", acc)