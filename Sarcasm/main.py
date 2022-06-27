from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,LSTM,Embedding, Flatten
import time
import data_preprocessing as x
from sklearn.metrics import classification_report
import numpy as np
x_train,y_train,x_test,y_test=x.preprocessing('Sarcasm_data.csv')

model=Sequential()
model.add(Embedding(input_dim = 2000, output_dim = 1024,input_length=100))
model.add(LSTM(128))
model.add(Dense(6, activation = "tanh"))
model.add(Dense(1,activation='sigmoid'))


model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
model.summary()
start=time.time()
model.fit(x_train,y_train,epochs=10,batch_size=64)
end=time.time()
print('training time was:',round(end-start),' seconds')
result=(model.predict(x_test))
for x in range(len(x_test)):
    result[x] = np.round(result[x][0])
print(classification_report(y_test,result))

model.save('Sarcasm.h5')