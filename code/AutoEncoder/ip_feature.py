def ip_to_bin(ip):

    ip_ = ip.split('.')

    binary = []
    for i in ip_:
        ip_bin = bin(int(i))[2:]
        ip_bin = '0'*(8-len(ip_bin)) + ip_bin
        binary.append(np.array(list(ip_bin),dtype=int))

    return binary
import numpy as np
ip = ['192.168.0.6','203.246.112.71','203.246.112.74','192.168.0.2','192.168.1.2','192.168.1.4','222.245.123.2','222.245.123.4']
cls = [0,1,1,0,2,2,3,3]
cls = [np.array([i]) for i in cls]
X = []
for ip_ in ip:
    re = ip_to_bin(ip_)
    # print(np.array(re).shape)
    X.append(np.array(re))
X = np.array(X)
# print(X.shape)
import tensorflow as tf
from tensorflow.keras.layers import *

from tensorflow.keras.models import Sequential

model = Sequential()
# print(X.shape)
model.add(
    Input(shape=(4,8,))
)
model.add(
    Flatten()
)
model.add(
    Dense(20)
)
model.add(
    Dense(4,activation='softmax')
)

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
cls = np.array(cls)
model.fit(X,cls,batch_size=1,epochs=30)

#0
ip__ = ip_to_bin('192.168.0.3')
test = np.array([ip__])
result = model.predict(test)
print(tf.argmax(result,axis=-1).numpy())

#2
ip__ = ip_to_bin('192.168.1.7')
test = np.array([ip__])
result = model.predict(test)
print(tf.argmax(result,axis=-1).numpy())

#3
ip__ = ip_to_bin('222.245.123.7')
test = np.array([ip__])
result = model.predict(test)
print(tf.argmax(result,axis=-1).numpy())

#1
ip__ = ip_to_bin('203.246.112.77')
test = np.array([ip__])
result = model.predict(test)
print(tf.argmax(result,axis=-1).numpy())