import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split #트레인, 테스트를 잘라주는 기능이 들어있는 함수

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
# y = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,7,5,7,8,6,10])

#[검색] train과 test를 섞어서 7:3 나눈다
# 힌트: 사이킷런

x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    # train_size=0.8, #오버플로우 (0.8+0.3 = 1이 넘어감)
    train_size=0.7,
    # test_size=0.3, # 트레인, 테스트 사이즈 중에 하나 생략(주석)해도 결과가 같음
    # shuffle=True, # 디폴트는 섞기로 되어있으므로, 섞는게 의도면 생략 가능
    random_state=1004, # 난수에 맞는 숫자를 제공, 재실행해도 이 난수를 바탕으로 숫자를 빼줘서 같은 결과가 나옴 (결과값 고정->데이터 일관성) 
)
# train_size와 test_size를 모두 주석처리 해도 자동으로 자름
# 디폴트 값이 존재함 -> train 75%(0.75), test 25%(0.25) 

print('x_train : ', x_train)
print('x_test : ', x_test)
print('y_train : ', y_train)
print('y_test : ', y_test)
# 섞으라는 말을 하지 않아도 섞여서 나옴 / 섞는게 디폴트값임 (shuffle=True)

# 실행할 때마다 결과가 다르게 나옴 
# 값을 고정 시켜주려면? -> random_state(랜덤 난수표)

#2. 모델구성
model = Sequential()
model.add(Dense(3,input_dim=1))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500 ,batch_size=2)

print("===================================")

#4. 평가, 예측
loss = model.evaluate(x_test, y_test) # evaluate 결과(프로그레스바)도 터미널에 뜸
print('loss = ', loss)
result = model.predict(x) # predict 결과(프로그레스바)도 터미널에 뜸
print(result)


# 그래프 그리기 (시각화)
import matplotlib.pyplot as plt
plt.scatter(x, y)   # 데이터 점 찍기
plt.plot(x, result, color='red') # 선 긋기
plt.show() # 결과를 그래프로 보여줌