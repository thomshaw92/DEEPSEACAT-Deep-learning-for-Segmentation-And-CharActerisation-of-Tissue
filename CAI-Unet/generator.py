#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 13:07:38 2019

@author: uqmtottr
"""

### Creating out own generator inspired from: https://github.com/ellisdg/3DUnetCNN/blob/master/unet3d/generator.py
import os
import copy
from random import shuffle
import itertools
import numpy as np
from .utils import pickle_dump, pickle_load
from .utils.patches import compute_patch_indices, get_random_nd_index, get_patch_from_3d_data
from .augument import augment_data, random_permutation_x_y 

# Function to get the generators
def get_train_and_val_generators(data_file, batch_size, n_labels, training_keys_file, validation_keys_file,
                                           data_split=0.8, overwrite=False, labels=None, augment=False,
                                           augment_flip=True, augment_distortion_factor=0.25,
                                           validation_batch_size=None, skip_blank=True, permute=False):

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    