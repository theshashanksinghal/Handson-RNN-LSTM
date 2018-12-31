						"""Shekespeare text generation"""
						
import sys
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

data=open("shakespeare.txt","r").read().lower()
characters=len(data)
vocab=set(data)
labeling={ch:i for i,ch in enumerate(sorted(list(vocab)))}
unlabelling={i:ch for i,ch in enumerate(sorted(list(vocab)))}
x=[]
y=[]
seq_length=100
for i in range(0,characters-seq_length,1):
    seqx=data[i:i+seq_length]
    seqy=data[i+seq_length]
    x.append([labeling[j] for j in seqx])
    y.append(labeling[seqy])
datax=np.reshape(x,(len(x),seq_length,1))
datax=datax/float(len(vocab))
datay=np_utils.to_categorical(y)
model=Sequential()
model.add(LSTM(256,input_shape=(seq_length,1),return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(len(vocab),activation="softmax"))
model.compile(loss="categorical_crossentropy",optimizer="adam")
model.summary()
fil="weights.{epoch:02d}-{loss:.2f}.hdf5"
chpt=ModelCheckpoint(fil,monitor="loss",verbose=1,mode="min",save_best_only=True)
call_lt=[chpt]
model.fit(datax,datay,epochs=15,batch_size=64,callbacks=call_lt)

weight_file="weights.14-1.81.hdf5"
model.load_weights(weight_file)
model.compile(loss="categorical_crossentropy",optimizer="adam")
random_int=np.random.randint(0,len(x)-1)
rand_ip=x[random_int]
#s=np.reshape(pattern,(1,len(pattern),1))
p=''.join([unlabelling[i] for i in rand_ip])
print("\"",p,"\"")
for i in range(1000):
    s=np.reshape(rand_ip,(1,len(pattern),1))
    s=s/len(vocab)
    prediction=model.predict(s,verbose=0)
    index=np.argmax(prediction)
    output=unlabelling[index]
    rand_ip.append(index)
    sys.stdout.write(output)
    rand_ip=rand_ip[1:]
print("DONE")