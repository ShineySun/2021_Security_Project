from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Dropout, BatchNormalization, Reshape, LeakyReLU, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from autoencoder_model_dense import *
import tensorflow as tf
import numpy as np

def cos_sim(a, b): 
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))


import matplotlib.pyplot as plt

# %matplotlib inline

x_train = np.load('allow.npy')
allowtime = np.load('allowtime.npy')
# y_train = np.load('y_train.npy')
# x_train = x_train[:10000]
import pickle
import os

with open('datatimedist.pkl','rb') as f:
    datatime = pickle.load(f)

with open('vocab.pkl','rb') as f:
    vocab = pickle.load(f)
with open('country.pkl','rb') as f:
    ck = pickle.load(f)
    ck_i2w = {v:k for k,v in ck.items()}

if not os.path.exists('errorlist.pkl'):
    encoder_in = Input(shape=(12,))
    x = encoder(encoder_in)
    decoder_out = decoder(x)

    auto_encoder = Model(encoder_in, decoder_out)

    checkpoint_path = 'tmp/01-basic-auto-encoder-MNIST.ckpt'

    auto_encoder.load_weights(checkpoint_path)

    x_train = x_train/len(vocab.keys())
    xy = encoder.predict(x_train,verbose=1)

    print(xy.shape)

    result = decoder.predict(xy,verbose=1)
    # result = tf.argmax(result,axis=2)
    y = []
    yy = []
    print(result[0].shape)
    mse = tf.keras.losses.MeanSquaredError()
    from tqdm import tqdm
    for i in tqdm(range(len(x_train))):
        x = x_train[i]
        re = result[i]
        error = mse(x,re).numpy()
        # yy.append(error)
        y.append(error)
    with open('errorlist.pkl','wb') as f:
        pickle.dump(y,f)
        # if error > 0.18:
        #     y.append(error)
        #     # print(x[:8]*255)
        #     # print(x[8:8+2]*65535)
        #     ct = x[10:10+2]*242
        #     # print(ct)
        #     yy.append([1])
        #     print(ck_i2w[int(ct[0])],ck_i2w[int(ct[1])])
        #     # y.append([1])
        # else:
        #     yy.append([0])
        # x = np.reshape(x,(8*8*2))
        # print(re)
        # print(re)
        # re = np.round(re,0)
        # print(re)
        # print(re)
        # print(re)
        # re = np.reshape(re,(8*8*2))
        # sim = cos_sim(x,re)
        # sim = x - re
        # print(sim)
        # print(x,re)
        # print(np.sum(sim)/(8*8*2))
        # if np.sum(sim)/(8*8*2) > 0.57:
        #     y.append(np.array([1]))
        # else:
        #     y.append(np.array([0]))
        # y.append(np.array([sim]))
else:
    with open('errorlist.pkl','rb') as f:
        y = pickle.load(f)  
        temp = []

        #     y.append(error)
        #     # print(x[:8]*255)
        #     # print(x[8:8+2]*65535)
        #     ct = x[10:10+2]*242
        #     # print(ct)
        #     yy.append([1])
        #     print(ck_i2w[int(ct[0])],ck_i2w[int(ct[1])])

        for index,yy in enumerate(y):
            if yy > 0.17:
                temp.append(yy)
                x = x_train[index] 
                ct = x[10:10+2]*242
                    # #     # print(ct)
                    # #     yy.append([1])
                print(x[0:4]*255)
                print(x[4:8]*255)
                print(x[8:10]*65535)
                print(ck_i2w[int(ct[0])],ck_i2w[int(ct[1])])
                classip = list(x[0:3]*255)
                classip = [str(int(i)) for i in classip]
                # print(classip)
                classip = '.'.join(classip)
                # # print(str(allowtime[index])[:12])
                # # print(str(allowtime[index])[:12]+classip)
                # # print(allowtime[index])
                # # print(datatime[str(allowtime[index])[:-4]+classip])
                print("")
        y = temp
plt.figure(figsize=(15, 12))
# plt.scatter(x=xy[:, 0], y=xy[:, 1], c=yy, cmap=plt.get_cmap('Paired'), s=3)
# plt.scatter(x=xy[:, 0], y=xy[:, 1], cmap=plt.get_cmap('Paired'), s=3)
# plt.colorbar()
plt.plot(range(len(y)),y)
plt.xlabel('xAxis name')
plt.ylabel('yAxis name')
plt.show()