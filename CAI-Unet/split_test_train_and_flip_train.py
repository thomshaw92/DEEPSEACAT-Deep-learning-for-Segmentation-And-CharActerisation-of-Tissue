#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 11:09:44 2019

@author: uqmtottr
"""
import os
from glob import glob
import random
import shutil
import numpy as np
import nibabel as nib

data_path = '/winmounts/uqmtottr/uq-research/DEEPSEACAT-Q1219/data/data_for_network/'

tse_path = sorted(glob(data_path+'tse/*.nii.gz'))
mprage_path= sorted(glob(data_path+'mprage/*.nii.gz'))
seg_path= sorted(glob(data_path+'seg/*.nii.gz'))

test_tse_path = '/winmounts/uqmtottr/uq-research/DEEPSEACAT-Q1219/data/data_for_network/test/tse/'
test_mprage_path = '/winmounts/uqmtottr/uq-research/DEEPSEACAT-Q1219/data/data_for_network/test/mprage/'
test_seg_path = '/winmounts/uqmtottr/uq-research/DEEPSEACAT-Q1219/data/data_for_network/test/seg/'

###########
# Step 1 #
###########

# Split data into a test and train set (the train set will be split into train and val in the generator later on)
# 10% of the dataset will be extracted for test (10% of 111 = 11)
# First we define which positions in the 11 test subjects have
total_n_sub = list(range(len(tse_path))) #The lenght of tse, mprage and seg paths are the same, so here we just use tse
n_test_sub = 11
testset_sub = random.sample(total_n_sub, n_test_sub)
testset_sub = sorted(testset_sub)

# The data paths of the data for the final test set will be saved in these three lists
test_tse_addrs = []
test_mprage_addrs = []
test_seg_addrs = []

# The data paths of the remaining data for training and validation will be saved in these three lists
train_tse_addrs = []
train_mprage_addrs = []
train_seg_addrs = []

# Now we extract the test data into the test lists and the remaning data into the test lists
for i in range(len(tse_path)):
    if i in testset_sub:
        test_tse_addrs.append(tse_path[i])
        test_mprage_addrs.append(mprage_path[i])
        test_seg_addrs.append(seg_path[i])
    else:
        train_tse_addrs.append(tse_path[i])
        train_mprage_addrs.append(mprage_path[i])
        train_seg_addrs.append(seg_path[i])


for k in range(len(test_tse_addrs)):
    shutil.move(test_tse_addrs[k], test_tse_path)
    shutil.move(test_mprage_addrs[k], test_mprage_path)
    shutil.move(test_seg_addrs[k], test_seg_path)



##########
# Step 2 #
##########

# Flip the data in the training list as a data augmentation
# The flip will be left right flip in the z-axis of the images
for j in range(len(train_tse_addrs)):
    load_tse = nib.load(train_tse_addrs[j])
    load_mprage = nib.load(train_mprage_addrs[j])
    load_seg = nib.load(train_seg_addrs[j])
    
    get_tse = load_tse.get_fdata()
    get_mprage = load_mprage.get_fdata()
    get_seg = load_seg.get_fdata()
    
    arr_tse = np.array(get_tse)
    arr_mprage = np.array(get_mprage)    
    arr_seg = np.array(get_seg)
    
    flipped_tse = arr_tse[:, :, ::-1]
    flipped_mprage = arr_mprage[:, :, ::-1]
    flipped_seg = arr_seg[:, :, ::-1]
    
    nifti_flipped_tse = nib.Nifti1Image(flipped_tse, affine=None)
    nifti_flipped_mprage = nib.Nifti1Image(flipped_mprage, affine=None)
    nifti_flipped_seg = nib.Nifti1Image(flipped_seg, affine=None)
    
    filename_flipped_tse = 'flipped_' + train_tse_addrs[j][train_tse_addrs[j].index('tse')+4:len(train_tse_addrs[j])]
    filename_flipped_mprage = 'flipped_' + train_mprage_addrs[j][train_mprage_addrs[j].index('mprage')+7:len(train_mprage_addrs[j])]    
    filename_flipped_seg = 'flipped_' + train_seg_addrs[j][train_seg_addrs[j].index('seg')+4:len(train_seg_addrs[j])]
    
    nib.save(nifti_flipped_tse, os.path.join(data_path + 'tse/' + filename_flipped_tse))
    nib.save(nifti_flipped_mprage, os.path.join(data_path + 'mprage/' + filename_flipped_mprage))
    nib.save(nifti_flipped_seg, os.path.join(data_path + 'seg/' + filename_flipped_seg))

'''    
img = '/winmounts/uqmtottr/uq-research/DEEPSEACAT-Q1219/data/ashs_atlas_umcutrecht/train/train000/tse_native_chunk_left.nii.gz'
flip_path = '/winmounts/uqmtottr/uq-research/DEEPSEACAT-Q1219/'

load_img = nib.load(img)
get_img = load_img.get_fdata()
img_arr = np.array(get_img)

flipped = img_arr[:, :, ::-1]

flip_img = nib.Nifti1Image(flipped, affine=None)
lab_file = 'flipped_image.nii.gz'
nib.save(flip_img, os.path.join(dir_path+lab_file))
'''
