from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Dropout, BatchNormalization, Reshape, LeakyReLU, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from autoencoder_model_dense import *
import tensorflow as tf

# # encoder_input = Input(shape=(28, 28, 1))
# encoder_input = Input(shape=(8, 8, 1))

# # 28 X 28
# x = Conv2D(32, 3, padding='same')(encoder_input) 
# x = BatchNormalization()(x)
# x = LeakyReLU()(x) 

# # 28 X 28 -> 14 X 14
# x = Conv2D(64, 3, strides=2, padding='same')(x)
# x = BatchNormalization()(x) 
# x = LeakyReLU()(x) 

# # 14 X 14 -> 7 X 7
# x = Conv2D(64, 3, strides=2, padding='same')(x)
# x = BatchNormalization()(x)
# x = LeakyReLU()(x)

# # 17 X 7
# x = Conv2D(64, 3, padding='same')(x)
# x = BatchNormalization()(x)
# x = LeakyReLU()(x)

# x = Flatten()(x)

# # 2D 좌표로 표기하기 위하여 2를 출력값으로 지정합니다.
# encoder_output = Dense(2)(x)

# encoder = Model(encoder_input, encoder_output)

# encoder.summary()
# # import sys
# # sys.exit()
# # Input으로는 2D 좌표가 들어갑니다.
# decoder_input = Input(shape=(2, ))

# # 2D 좌표를 7*7*64 개의 neuron 출력 값을 가지도록 변경합니다.



# x = Reshape( (2, 2, 64))(x)

# # 7 X 7 -> 7 X 7
# x = Conv2DTranspose(64, 3, strides=1, padding='same')(x)
# x = BatchNormalization()(x)
# x = LeakyReLU()(x)

# # 7 X 7 -> 14 X 



# x = Conv2DTranspose(64, 3, strides=2, padding='same')(x)
# x = BatchNormalization()(x)
# x = LeakyReLU()(x)

# # 14 X 14 -> 28 X 28
# x = Conv2DTranspose(64, 3, strides=2, padding='same')(x)
# x = BatchNormalization()(x)
# x = LeakyReLU()(x)

# # 28 X 28 -> 28 X 28
# x = Conv2DTranspose(32, 3, strides=1, padding='same')(x)
# x = BatchNormalization()(x)
# x = LeakyReLU()(x)

# # 최종 output
# decoder_output = Conv2DTranspose(1, 3, strides=1, padding='same', activation='tanh')(x)

# decoder = Model(decoder_input, decoder_output)
# decoder.summary()
# # import sys
# # sys.exit()
import numpy as np
x_train = np.load('deny.npy')
# y_train = np.load('y_train.npy')
LEARNING_RATE = 0.0005
BATCH_SIZE = 100

encoder_in = Input(shape=(12,))

x = encoder(encoder_in)
decoder_out = decoder(x)

auto_encoder = Model(encoder_in, decoder_out)

auto_encoder.compile(optimizer=tf.keras.optimizers.Adam(),
loss=tf.keras.losses.MeanSquaredError()
,metrics=['accuracy'])

checkpoint_path = 'tmp/01-basic-auto-encoder-MNIST.ckpt'
# checkpoint_path = 'tmp_not_country/01-basic-auto-encoder-MNIST.ckpt'
checkpoint = ModelCheckpoint(checkpoint_path, 
                             save_best_only=True, 
                             save_weights_only=True, 
                             monitor='loss', 
                             verbose=1)

with open('vocab.pkl','rb') as f:
    vocab = pickle.load(f)
x_train = x_train/len(vocab.keys())
auto_encoder.fit(x_train, x_train, 
                 batch_size=BATCH_SIZE, 
                 epochs=10, 
                 callbacks=[checkpoint], 
                )
# print(x_train)
# print(tf.argmax(auto_encoder.predict(x_train)[0],axis=1))
# auto_encoder.load_weights(checkpoint_path)

# import matplotlib.pyplot as plt

# # %matplotlib inline

# xy = encoder.predict(x_train)

# print(xy.shape, y_train.shape)

# plt.figure(figsize=(15, 12))
# plt.scatter(x=xy[:, 0], y=xy[:, 1], c=y_train, cmap=plt.get_cmap('Paired'), s=3)
# plt.colorbar()
# plt.show()