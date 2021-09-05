# -*- coding: utf-8 -*-
"""
@author: angel
"""

import tensorflow as tf
from tensorflow.keras.layers import MaxPooling3D, Flatten, Dense, Conv3D, BatchNormalization, Input, Dropout, GlobalAveragePooling3D

def model_0(input_shape):
    
    inputs = tf.keras.layers.Input(input_shape)
    
    x = tf.keras.layers.Conv3D(filters=32, kernel_size=5, activation="relu")(inputs)
    x = MaxPooling3D(pool_size=2)(x)
    
    x = Flatten()(x)
    x = Dense(units=200, activation="relu")(x)
    
    x = Dropout(rate=0.1)(x)
    outputs = Dense(units=3, activation="softmax")(x)
   
    model = tf.keras.Model(inputs, outputs, name="model_0_mri")
    return model

def model_1(input_shape):
    
    inputs = tf.keras.layers.Input(input_shape)

    x = Conv3D(filters=16, kernel_size=5, activation="relu")(inputs)
    x = MaxPooling3D(pool_size=2)(x)

    x = Conv3D(filters=32, kernel_size=5, activation="relu")(x)
    x = MaxPooling3D(pool_size=2)(x)

    x = Conv3D(filters=64, kernel_size=5, activation="relu")(x)
    x = MaxPooling3D(pool_size=2)(x)

    x = Flatten()(x)
    x = Dropout(0.2)(x)
    x = Dense(units=256, activation="relu")(x)

    outputs = Dense(units=3, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="model_1_mri")
    return model

def model_2(input_shape):
    
    inputs = tf.keras.layers.Input(input_shape)

    x = Conv3D(filters=16, kernel_size=5, activation="relu")(inputs)
    x = MaxPooling3D(pool_size=2)(x)

    x = Conv3D(filters=32, kernel_size=5, activation="relu")(x)
    x = MaxPooling3D(pool_size=2)(x)

    x = Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = Conv3D(filters=64, kernel_size=3, activation="relu")(x)

    x = MaxPooling3D(pool_size=2)(x)

    x = Flatten()(x)
    x = Dropout(0.25)(x)
    x = Dense(units=256, activation="relu")(x)

    outputs = Dense(units=3, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="model_2_mri")
    return model



def model_3(input_shape):
    
    inputs = tf.keras.layers.Input(input_shape)

    x = Conv3D(filters=16, kernel_size=3, activation='relu')(inputs)
    x = Conv3D(filters=16, kernel_size=3, activation='relu')(x)
    x = MaxPooling3D(pool_size=2)(x)

    x = Conv3D(filters=32, kernel_size=3, activation="relu")(x)
    x = Conv3D(filters=32, kernel_size=3, activation='relu')(x)
    x = MaxPooling3D(pool_size=2)(x)

    x = BatchNormalization(momentum=0.9)(x)
    x = Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = Conv3D(filters=64, kernel_size=3, activation='relu')(x)
    x = MaxPooling3D(pool_size=2)(x)

    x = Flatten()(x)
    x = Dropout(rate=0.2)(x)
    x = Dense(units=256, activation="relu")(x)
    x = Dropout(rate=0.1)(x)
    x = Dense(units=128, activation='relu')(x)

    outputs = Dense(units=3, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="model_3_mri")
    return model


def model_4(input_shape):
    
    inputs = Input(input_shape)

    x = Conv3D(filters=32, kernel_size=3, activation='relu')(inputs)
    x = Conv3D(filters=32, kernel_size=3, activation='relu')(x) 
    x = MaxPooling3D(pool_size=2)(x)

    x = Conv3D(filters=64, kernel_size=3, activation='relu')(x)
    x = Conv3D(filters=64, kernel_size=3, activation='relu')(x)
    x = MaxPooling3D(pool_size=2)(x)

    x = BatchNormalization(momentum=0.9)(x)
    x = Conv3D(filters=128, kernel_size=3, activation='relu')(x)
    x = Conv3D(filters=128, kernel_size=3, activation='relu')(x)
    x = MaxPooling3D(pool_size=2)(x)

    x = BatchNormalization(momentum=0.9)(x)
    x = Conv3D(filters=256, kernel_size=3, activation='relu')(x)
    x = Conv3D(filters=256, kernel_size=3, activation='relu')(x)
    x = Conv3D(filters=256, kernel_size=3, activation='relu')(x)
    x = GlobalAveragePooling3D()(x)

    x = Dropout(rate=0.2)(x)
    x = Dense(units=128, activation='relu')(x)
    x = Dropout(rate=0.2)(x)
    x = Dense(units=128, activation='relu')(x)

    outputs = Dense(units=3, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="model_4_mri")
    return model