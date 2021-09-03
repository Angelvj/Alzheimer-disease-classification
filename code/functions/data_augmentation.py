# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 12:31:56 2021

@author: angel
"""

import skimage.transform as transform
import tensorflow as tf
import scipy
import numpy as np


def augment_image(img):
    img = img.squeeze()
    original_shape = img.shape
    img = random_rotations(img, -5, 5)
    img = random_zoom(img, min=0.9, max=1.1)
    img = random_shift(img, max=0.2)
    # img = random_flip(img)
    img = downscale(img, original_shape)
    img = np.expand_dims(img, axis=3) # Restore channel axis
    return img


def downscale(image, shape):
    'For upscale, anti_aliasing should be false'
    return transform.resize(image, shape, mode='constant', anti_aliasing=True)


@tf.function(input_signature=[tf.TensorSpec(None, tf.float32)])
def tf_augment_image(input):
    """ Tensorflow can't manage numpy functions, we have to wrap our augmentation function """
    img = tf.numpy_function(augment_image, [input], tf.float32)
    return img


def random_rotations(img, min_angle, max_angle):
    """
    Rotate 3D image randomly
    """
    assert img.ndim == 3, "Image must be 3D"
    rotation_axes = [(1, 0), (1, 2), (0, 2)]
    angle = np.random.randint(low=min_angle, high=max_angle+1)
    axes_random_id = np.random.randint(low=0, high=len(rotation_axes))
    axis = rotation_axes[axes_random_id] # Select a random rotation axis
    return scipy.ndimage.rotate(img, angle, axes=axis)


def random_zoom(img,min=0.7, max=1.2):
    """
    Generate random zoom of a 3D image
    """
    zoom = np.random.sample()*(max - min) + min # Generate random zoom between min and max
    zoom_matrix = np.array([[zoom, 0, 0, 0],
                            [0, zoom, 0, 0],
                            [0, 0, zoom, 0],
                            [0, 0, 0, 1]])
    
    return scipy.ndimage.interpolation.affine_transform(img, zoom_matrix)


def random_flip(img):
    """
    Flip image over a random axis
    """
    axes = [0, 1, 2]
    rand_axis = np.random.randint(len(axes))
    img = img.swapaxes(rand_axis, 0)
    img = img[::-1, ...]
    img = img.swapaxes(0, rand_axis)
    img = np.squeeze(img)
    return img


def random_shift(img, max=0.4):
    """
    Random shift over a random axis
    """
    (x, y, z) = img.shape
    (max_shift_x, max_shift_y, max_shift_z) = int(x*max/2),int(y*max/2), int(z*max/2)
    shift_x = np.random.randint(-max_shift_x, max_shift_x)
    shift_y = np.random.randint(-max_shift_y,max_shift_y)
    shift_z = np.random.randint(-max_shift_z,max_shift_z)

    translation_matrix = np.array([[1, 0, 0, shift_x],
                                   [0, 1, 0, shift_y],
                                   [0, 0, 1, shift_z],
                                   [0, 0, 0, 1]
                                   ])

    return scipy.ndimage.interpolation.affine_transform(img, translation_matrix)