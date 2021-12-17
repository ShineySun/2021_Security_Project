from tensorflow.keras.layers import Conv1D,Conv2D, Conv2DTranspose,Conv1DTranspose, Dense, Flatten, Dropout, BatchNormalization, Reshape, LeakyReLU, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
import pickle
with open('vocab.pkl','rb') as f:
    vocab = pickle.load(f)
# encoder_input = Input(shape=(28, 28, 1))
encoder_input = Input(shape=(12,1))

# 28 X 28
x = Conv1D(10, 3, padding='same')(encoder_input) 
x = BatchNormalization()(x)
x = LeakyReLU()(x) 

# 28 X 28 -> 14 X 14
x = Conv1D(5, 3, strides=2, padding='same')(x)
x = BatchNormalization()(x) 
x = LeakyReLU()(x) 

# 14 X 14 -> 7 X 7
# x = Conv2D(64, 3, strides=2, padding='same')(x)
# x = BatchNormalization()(x)
# x = LeakyReLU()(x)

# 17 X 7
# x = Conv2D(64, 3, padding='same')(x)
# x = BatchNormalization()(x)
# x = LeakyReLU()(x)

x = Flatten()(x)

# 2D 좌표로 표기하기 위하여 2를 출력값으로 지정합니다.
encoder_output = Dense(2)(x)

encoder = Model(encoder_input, encoder_output)

encoder.summary()
# import sys
# sys.exit()
# Input으로는 2D 좌표가 들어갑니다.
decoder_input = Input(shape=(2, ))

# 2D 좌표를 7*7*64 개의 neuron 출력 값을 가지도록 변경합니다.
x = Dense(6*5)(decoder_input)
x = Reshape( (6, 5))(x)

# 7 X 7 -> 7 X 7
x = Conv1DTranspose(5, 3, strides=1, padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

# 7 X 7 -> 14 X 
x = Conv1DTranspose(10, 3, strides=2, padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

# 14 X 14 -> 28 X 28
# x = Conv2DTranspose(64, 3, strides=2, padding='same')(x)
# x = BatchNormalization()(x)
# x = LeakyReLU()(x)

# # 28 X 28 -> 28 X 28
# x = Conv2DTranspose(32, 3, strides=1, padding='same')(x)
# x = BatchNormalization()(x)
# x = LeakyReLU()(x)

# 최종 output
decoder_output = Conv1DTranspose(1, 3, strides=1, padding='same', activation='tanh')(x)
# decoder_output = x
decoder = Model(decoder_input, decoder_output)
decoder.summary()
# import sys
# sys.exit()