#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 11:25:27 2019

@author: uqmtottr
"""

# Function to convert the input MR images and segmentations to hpf5 files.

import h5py
import numpy as np

#umc_tse = 
#umc_mprage = 
#umc_seg = 
arr = r'C:/Users/uqmtottr/Desktop/Images_Test_Generator/train/tse/'

with h5py.File('random.hdf5', 'w') as f:
    dset = f.create_dataset("default", data=arr)
    

#Read the data in the .hpf5 file
with h5py.File('random.hdf5', 'r') as f:
    data = f['default']
    print(min(data))
    print(max(data))
    print(data[:15])


'''
import os
import tables

def create_data_file(out_file, n_channels, n_samples, image_shape):
    hdf5_file = tables.open_file(out_file, mode='w')
    filters = tables.Filters(complevel=5, complib='blosc')
    data_shape = tuple([0, n_channels] + list(image_shape))
    truth_shape = tuple([0, 1] + list(image_shape))
    data_storage = hdf5_file.create_earray(hdf5_file.root, 'data', tables.Float32Atom(), shape=data_shape,
                                           filters=filters, expectedrows=n_samples)
    truth_storage = hdf5_file.create_earray(hdf5_file.root, 'truth', tables.UInt8Atom(), shape=truth_shape,
                                            filters=filters, expectedrows=n_samples)
    return hdf5_file, data_storage, truth_storage

def write_data_to_file(training_data_files, out_file, image_shape, subject_ids=None):
    n_samples = len(training_data_files)
    n_channels = len(training_data_files[0]) - 1
    
    try:
        hdf5_file, data_storage, truth_storage = create_data_file(out_file,
                                                                  n_channels=n_channels,
                                                                  n_sample=n_samples,
                                                                  image_shape=image_shape)
    except Exception as e:
        # If something goes wrong, delete the incomplete data file 
        os.remove(out_file)
        raise e
            
    if subject_ids:
        hdf5_file.create_array(hdf5_file.root, 'subject_ids', obj=subject_ids)
    hdf5_file.close()
    return out_file

def open_data_file(filename, readwrite="r"):
    return tables.open_file(filename, readwrite)
    '''