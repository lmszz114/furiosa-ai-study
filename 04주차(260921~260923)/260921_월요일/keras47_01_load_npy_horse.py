import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D
import time
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split



#1. 데이터

np_path = 'H:/furiosa-ai-study/04주차(260921~260923)/260921_월요일/_save/'
x_train = np.load(np_path + "horse_x_train.npy")
y_train = np.load(np_path + "horse_y_train.npy")
x_test = np.load(np_path + "horse_x_test.npy")
y_test = np.load(np_path + "horse_y_test.npy")


#2. 모델구성
model = Sequential()
model.add(Conv2D(64, (7,7), input_shape=(200, 200, 3),))  
model.add(Dropout(0.2))
model.add(MaxPooling2D())
model.add(Conv2D(32, (7,7), activation='relu',)) 
model.add(Dropout(0.2))
model.add(MaxPooling2D())
model.add(Conv2D(16, (5,5), activation='relu',)) 
model.add(Dropout(0.2))
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
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


es = EarlyStopping(
    monitor='val_acc',
    mode='auto',
    patience=50,
    restore_best_weights=True
    )
mcp = ModelCheckpoint(monitor='val_acc', mode='auto', save_best_only=True, filepath=np_path + 'horse_human.keras', verbose=1,)

start_time = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=64,
          verbose=1, 
          validation_split=0.2,
          callbacks=[es,mcp],
          )
end_time = time.time()


#4. 평가, 예측
print("==================model.evaluate==================")
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss: ', loss[0])
print('acc: ', loss[1])

y_predict = model.predict(x_test)
y_predict = np.round(y_predict) 

acc_score = accuracy_score(y_test, y_predict)
print("acc: ", round(acc_score, 4))
print("훈련 시간: ", round(end_time - start_time, 2), "초")

"""
loss:  0.03686167299747467
acc:  0.9902912378311157
7/7 [==============================] - 0s 14ms/step
acc:  0.9903
훈련 시간:  122.45 초
"""