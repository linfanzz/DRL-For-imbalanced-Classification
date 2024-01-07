# coding=utf-8
import tensorflow as tf
import os
import numpy as np

AUTOTUNE = tf.data.experimental.AUTOTUNE
MAX_PKT_BYTES = 50 * 50
MAX_PKT_NUM = 100
_batch_size = 1 #TODO(zenglinfan) batch train

_pkt_num = 20
_pkt_bytes = 256
_num_classes = 15

def _parse_sparse_example(example_proto):
    features = {
        'sparse': tf.io.SparseFeature(index_key=['idx1', 'idx2'],
                                        value_key='val',
                                        dtype=tf.int64,
                                        size=[MAX_PKT_NUM, MAX_PKT_BYTES]),
        'label': tf.io.FixedLenFeature([], dtype=tf.int64),
        'byte_len': tf.io.FixedLenFeature([], dtype=tf.int64),
        'last_time': tf.io.FixedLenFeature([], dtype=tf.float32),
    }
    batch_sample = tf.io.parse_example(example_proto, features)
    sparse_features = batch_sample['sparse']
    labels = batch_sample['label']
    sparse_features = tf.sparse.slice(sparse_features, start=[0, 0], size=[_pkt_num, _pkt_bytes])
    dense_features = tf.sparse.to_dense(sparse_features)
    dense_features = tf.cast(dense_features, tf.float32) / 255. # type: ignore
    return dense_features, labels


def generate_ds(path_dir, use_cache=False):
    assert os.path.isdir(path_dir)
    ds = tf.data.Dataset.list_files(os.path.join(path_dir, '*.tfrecord'), shuffle=True)
    ds = ds.interleave(
        lambda x: tf.data.TFRecordDataset(x).map(_parse_sparse_example),
        cycle_length=AUTOTUNE, 
        block_length=8,
        num_parallel_calls=AUTOTUNE
    ) # type: ignore
    # ds = ds.batch(_batch_size, drop_remainder=False)
    if use_cache:
        ds = ds.cache()
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

