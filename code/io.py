# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 17:46:03 2021

@author: angel
"""

import numpy as np
import tensorflow as tf
import nibabel as nib
import config as cfg

def load_image(path):    
    """
    Parameters
    ----------
    path : str
        absolute path to the image

    Returns
    -------
    img : numpy ndarray
        array containing the image
    """
    img = nib.load(path)
    img = np.asarray(img.dataobj)
    img = np.expand_dims(img, axis=3) # Add dummy axis for channel
    return img

def get_gcs_path(dataset):
    from kaggle_datasets import KaggleDatasets
    from kaggle_secrets import UserSecretsClient
    
    user_secrets = UserSecretsClient()
    user_credential = user_secrets.get_gcloud_credential()
    
    user_secrets.set_tensorflow_credential(user_credential)
    
    gcs_ds_path = KaggleDatasets().get_gcs_path(dataset)
    
    return gcs_ds_path

def read_tfrecord(example):
    tfrec_format = {
        "image": tf.io.FixedLenSequenceFeature(shape=(), dtype=tf.float32, allow_missing=True),
#         "image": tf.io.VarLenFeature(tf.float32),
        "label": tf.io.FixedLenFeature([], tf.int64),
        "one_hot_label": tf.io.VarLenFeature(tf.float32),
        "shape": tf.io.FixedLenFeature([4], tf.int64),
        "filename": tf.io.FixedLenFeature([], tf.string) # Only for test, TODO: delete
    }

    example = tf.io.parse_single_example(example, tfrec_format)
    one_hot_label = tf.sparse.to_dense(example['one_hot_label'])
    one_hot_label = tf.reshape(one_hot_label, [3])
    image  = tf.reshape(example['image'], example['shape'])
#     label = example['label']

    return image, one_hot_label

def load_dataset(filenames):
    
    # Allow order-altering optimizations
    option_no_order = tf.data.Options()
    option_no_order.experimental_deterministic = False
    
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads = cfg.AUTO)
    dataset = dataset.with_options(option_no_order)
    dataset = dataset.map(read_tfrecord, num_parallel_calls = cfg.AUTO)
    return dataset

def get_batched_dataset(filenames, batch_size = 4, train=False, augment=True, cache=True, shuffle_buff_size = 2048):
    dataset =  load_dataset(filenames)
    if cache:
        dataset = dataset.cache() # Only if dataset fits in ram
    if train:
        dataset = dataset.repeat()
        if augment:
            raise NotImplementedError
#             dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
        dataset = dataset.shuffle(shuffle_buff_size) # Not for shure
    dataset = dataset.batch(batch_size * cfg.REPLICAS)
    dataset = dataset.prefetch(cfg.AUTO)
    return dataset