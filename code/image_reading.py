# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 19:47:44 2021

@author: angel
"""

import nibabel as nib
import numpy as np

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
