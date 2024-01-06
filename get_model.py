#coding=utf-8
import keras
import tensorflow as tf
import numpy as np
from keras.models import Model, Sequential
from keras import layers
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Activation, Embedding
from keras.optimizers import Adam, SGD
from keras.layers import LSTM

def get_text_model(input_shape,output):
    top_words, max_words = input_shape
    model = Sequential()
    model.add(Embedding(top_words, 128, input_length=max_words))
    model.add(Flatten())
    model.add(Dense(250))
    model.add(Activation('relu'))
    model.add(Dense(output))
    return model


def get_image_model(in_shape, output):
    model = Sequential()
    model.add(Conv2D(32, (5, 5), padding='Same', input_shape=in_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (5, 5), padding='Same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(output))
    return model

# pbcnn param
_pkt_num = 20
_pkt_bytes = 256
_num_class = 15

def _text_cnn_block(x, filters, height, width, data_format='channels_last'):
    x = layers.Conv2D(filters=filters, kernel_size=(height, width),
                        strides=1, data_format=data_format)(x)
    x = layers.BatchNormalization(axis=-1 if data_format == 'channels_last' else 1)(x)
    x = layers.Activation(activation='relu')(x)
    x = tf.reduce_max(x, axis=1, keepdims=False)
    return x

def get_pbcnn_model():
    x = Input(shape=(_pkt_num, _pkt_bytes, 1))
    # y = tf.reshape(x, shape=(-1, _pkt_num, _pkt_bytes, 1))
    y = tf.reshape(x, shape=(-1, _pkt_num, _pkt_bytes, 1))
    data_format = 'channels_last'
    y1 = _text_cnn_block(y, filters=256, height=3, width=_pkt_bytes)
    y2 = _text_cnn_block(y, filters=256, height=4, width=_pkt_bytes)
    y3 = _text_cnn_block(y, filters=256, height=5, width=_pkt_bytes)
    y = layers.concatenate(inputs=[y1, y2, y3], axis=-1)
    y = layers.Flatten(data_format=data_format)(y)
    y = layers.Dense(512, activation='relu')(y)
    y = layers.Dense(256, activation='relu')(y)
    # y = layers.Dense(128, activation='relu')(y)
    y = layers.Dense(_num_class, activation='linear')(y)
    return Model(inputs=x, outputs=y)
