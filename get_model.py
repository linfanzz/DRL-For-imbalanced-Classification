#coding=utf-8
import tensorflow as tf
from keras.models import Model
from keras import layers

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
    x = layers.Input(shape=(_pkt_num, _pkt_bytes, 1))
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
