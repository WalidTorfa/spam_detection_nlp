from tensorflow.keras.layers import Dense,Dropout,LSTM,SimpleRNN,Embedding
from tensorflow.keras.models import Sequential
from sklearn.metrics import classification_report
import data_preprocess as x
import numpy as np
import time


(x_test,x_train,y_train,y_test)=x.PreProcessing("spam_or_not_spam.csv")

model=Sequential()
model.add(Embedding(input_dim = 2000, output_dim = 1024,input_length=100))
model.add(LSTM(128))
model.add(Dense(6, activation = "tanh"))
model.add(Dense(6,activation='relu'))
model.add(Dense(1,activation='sigmoid'))


model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
tic=time.time()
model.fit(x_train,y_train,epochs=10,batch_size=64)
toc=time.time()
print('training time was:',str(toc-tic))

print(model.summary())
model.save("mymodel.h5")
result=(model.predict(x_test))
for x in range(len(x_test)):
    result[x] = np.round(result[x][0])
print(classification_report(y_test,result,zero_division=0))
from sklearn.metrics import roc_auc_score
lstm_auc=roc_auc_score(y_test,result)
print('auc for lstm=',lstm_auc)