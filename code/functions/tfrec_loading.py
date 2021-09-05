# -*- coding: utf-8 -*-
"""
@author: angel
"""


import tensorflow as tf
import re
import numpy as np

def read_tfrecord(example, img_shape, num_classes):
    
    tfrec_format = {'image': tf.io.VarLenFeature(tf.float32),
                    'one_hot_label': tf.io.VarLenFeature(tf.float32)
                    }
    
    example = tf.io.parse_single_example(example, tfrec_format)
    one_hot_label = tf.sparse.to_dense(example['one_hot_label'])
    one_hot_label = tf.reshape(one_hot_label, [num_classes])
    image = tf.reshape(tf.sparse.to_dense(example['image']), img_shape)
    
    return image, one_hot_label


def load_dataset(filenames, img_shape, num_classes, autotune, no_order=True):
    option_no_order = tf.data.Options()
    option_no_order.experimental_deterministic = False
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=autotune)
    if no_order:
        dataset = dataset.with_options(option_no_order)
    
    dataset = dataset.map(lambda example: read_tfrecord(example, img_shape, num_classes), 
                          num_parallel_calls=autotune)
    
    return dataset


def get_dataset(filenames, img_shape, num_classes, autotune, batch_size = 4, 
                train=False, augment=None, cache=False, no_order=True):

    dataset =  load_dataset(filenames, img_shape, num_classes, autotune, no_order)
    if cache:
        dataset = dataset.cache() # Do it only if dataset fits in ram
    if train:
        dataset = dataset.repeat()
        if augment is not None:
            dataset = dataset.map(lambda img, label: (augment(img), label), num_parallel_calls=autotune)

        dataset = dataset.shuffle(count_data_items(filenames))

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(autotune)
    return dataset


def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) 
            for filename in filenames]
    return np.sum(n)


