#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 11:25:27 2019

@author: uqmtottr
"""

import h5py
import numpy as np
import nibabel
from glob import glob
import random

# Addresses of the hdf5 output files (train/val and test)
hdf5_train_val = '/winmounts/uqmtottr/uq-research/DEEPSEACAT-Q1219/data/train_val_set.hdf5'
hdf5_test = '/winmounts/uqmtottr/uq-research/DEEPSEACAT-Q1219/data/testset_hdf5.hdf5'

# Addresses to data
data_path = '/data/fastertemp/uqmtottr/data_for_network/'

tse_path = glob(data_path+'tse/*.nii.gz')
mprage_path=glob(data_path+'mprage/*.nii.gz')
seg_path=glob(data_path+'seg/*.nii.gz')

# Empty lists, which will be used when we create the hdf5 ile and when we try to open the images in the .h5 files.
train_tse = []
train_mprage = []
train_seg = []

val_tse = []
val_mprage = []
val_seg = []

test_tse = []
test_mprage = []
test_seg = []

train_img = []
val_img = []
test_img = []


#### Extract 11 subjects from the total dataset for the testset and collect the remaining data in remain(tse,mprage,seg) lists. ####
total_n_sub = list(range(len(tse_path)))
n_test_sub = 11
testset_sub = random.sample(total_n_sub, n_test_sub)
testset_sub = sorted(testset_sub)

# The data paths of the data for the final test will be saved in these three lists
test_tse_addrs = []
test_mprage_addrs = []
test_seg_addrs = []

# The data paths of the remaining data for training and validation will be saved in these three lists and later on be split into train ad val sets
remain_tse_addrs = []
remain_mprage_addrs = []
remain_seg_addrs = []

for x in range(len(tse_path)):
    if x in testset_sub:
        test_tse_addrs.append(tse_path[x])
        test_mprage_addrs.append(mprage_path[x])
        test_seg_addrs.append(seg_path[x])
    else:
        remain_tse_addrs.append(tse_path[x])
        remain_mprage_addrs.append(mprage_path[x])
        remain_seg_addrs.append(seg_path[x])


#### Split the remaining data into training and validation ####
n_remain_sub = list(range(len(remain_tse_addrs)))
n_val_sub = 11
val_sub = random.sample(n_remain_sub, n_val_sub)
val_sub = sorted(val_sub)

# The data paths of the validation data
val_tse_addrs = []
val_mprage_addrs = []
val_seg_addrs = []

# The data paths of the training data
train_tse_addrs = []
train_mprage_addrs = []
train_seg_addrs = []

for y in range(len(remain_tse_addrs)):
    if y in val_sub:
        val_tse_addrs.append(remain_tse_addrs[y])
        val_mprage_addrs.append(remain_mprage_addrs[y])
        val_seg_addrs.append(remain_seg_addrs[y])
    
    else:
        train_tse_addrs.append(remain_tse_addrs[y])
        train_mprage_addrs.append(remain_mprage_addrs[y])
        train_seg_addrs.append(remain_seg_addrs[y])


#### Iterate over the train and val path lists to save the images in new lists ####
for i in range(len(train_tse_addrs)):
    train_tse.append(nibabel.load(train_tse_addrs[i]).get_fdata())
    train_mprage.append(nibabel.load(train_mprage_addrs[i]).get_fdata())
    train_seg.append(nibabel.load(train_seg_addrs[i]).get_fdata())

    if i < len(val_tse_addrs):
        val_tse.append(nibabel.load(val_tse_addrs[i]).get_fdata())
        val_mprage.append(nibabel.load(val_mprage_addrs[i]).get_fdata())
        val_seg.append(nibabel.load(val_seg_addrs[i]).get_fdata())
    else:
        pass

# Create the hdf5 file
with (h5py.File(hdf5_train_val, mode='w')) as hfile:
    # Creating the training group and its datasets and including data
    train_imagegroup = hfile.create_group('train')
    
    train_imagegroup.create_dataset("tse", data=train_tse)
    train_imagegroup.create_dataset("mprage", data=train_mprage)
    train_imagegroup.create_dataset("seg", data=train_seg)

    # Creating the validation group and its datasets and including data
    val_imagegroup = hfile.create_group('val')
            
    val_imagegroup.create_dataset("tse", data=val_tse)
    val_imagegroup.create_dataset("mprage", data=val_mprage)
    val_imagegroup.create_dataset("seg", data=val_seg)
    
hfile.close()           


#### Create a .h5 file for the test data and add data into three datasets (tse, mprage, seg) ####
for j in range(len(test_tse_addrs)):
        test_tse.append(np.array(nibabel.load(test_tse_addrs[j]).get_fdata()))
        test_mprage.append(nibabel.load(test_mprage_addrs[j]).get_fdata())
        test_seg.append(nibabel.load(test_seg_addrs[j]).get_fdata())
        
# Create the hdf5 file
with (h5py.File(hdf5_test, mode='w')) as testhfile:
    # Creating the three datasets and including data
    testhfile.create_dataset("tse", data=test_tse)
    testhfile.create_dataset("mprage", data=test_mprage)
    testhfile.create_dataset("seg", data=test_seg)
    
testhfile.close()