# -*- coding: utf-8 -*-
"""
@author: angel
"""

import tensorflow as tf

def get_lr_decay_callback(lr_max, lr_min, decay, verbose=False):
    def lrfn(epoch):
        lr = (lr_max - lr_min) * decay**(epoch) + lr_min
        return lr

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose)
    return lr_callback


def get_exp_lr_decay_callback(lr_ini, decay, epochs, verbose=False):
    def lrfn(epoch):
        lr = lr_ini * decay**(epoch/epochs)
        return lr

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose)
    return lr_callback