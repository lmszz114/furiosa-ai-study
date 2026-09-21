
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D
import time
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

#1. 데이터
path = 'H:/furiosa-ai-study/04주차(260921~260923)/260921_월요일/_save/'
x_train = np.load(path + "man_woman_x_train.npy")
y_train = np.load(path + "man_woman_y_train.npy")
x_test = np.load(path + "man_woman_x_test.npy")
y_test = np.load(path + "man_woman_y_test.npy")


#2. 모델구성
model = Sequential()
model.add(Conv2D(64, (7,7), input_shape=(150, 150, 3),))  
model.add(Dropout(0.2))
model.add(MaxPooling2D())
model.add(Conv2D(64, (7,7), activation='relu',))
model.add(Dropout(0.2))
model.add(MaxPooling2D()) 
model.add(Conv2D(32, (5,5), activation='relu',)) 
model.add(Dropout(0.2))
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))  


#3. 컴파일, 훈련
model.compile(loss="binary_crossentropy", 
              optimizer="adam", 
              metrics=['acc'],
              )

model_path = 'H:/furiosa-ai-study/04주차(260921~260923)/260921_월요일/_save/'
es = EarlyStopping(
    monitor='val_acc',
    mode='auto',
    patience=999,
    restore_best_weights=True
    )
mcp = ModelCheckpoint(monitor='val_acc', mode='auto', save_best_only=True, filepath=model_path + 'man_woman.keras', verbose=1,)

start_time = time.time()
model.fit(x_train, y_train, epochs=50, batch_size=128,
          verbose=1, 
          validation_split=0.2,
          callbacks=[es,mcp],
          )
end_time = time.time()

#4. 평가, 예측
print("==================model.evaluate==================")
loss = model.evaluate(x_test, y_test, verbose=1, batch_size=32)
print('loss: ', loss[0])
print('acc: ', loss[1])

y_predict = model.predict(x_test, batch_size=32)
y_predict = np.round(y_predict) 

acc_score = accuracy_score(y_test, y_predict)
print("acc: ", round(acc_score, 4))
print("훈련 시간: ", round(end_time - start_time, 2), "초")

"""
loss:  0.2437417358160019
acc:  0.9035701155662537
170/170 [==============================] - 1s 4ms/step
acc:  0.9036
훈련 시간:  305.6 초
"""