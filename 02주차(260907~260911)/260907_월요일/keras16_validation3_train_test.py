from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import numpy as np


#1. 데이터
x = np.array(range(1, 17))
y = np.array(range(1, 17))

# [실습] train_test_split 으로 잘라보기 (8, 4, 4)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.5, random_state=1111,)

x_val, x_test, y_val, y_test = train_test_split(
    x_test, y_test, test_size=0.5, random_state=1111,)

print(x_train)
print(x_val)
print(x_test)
