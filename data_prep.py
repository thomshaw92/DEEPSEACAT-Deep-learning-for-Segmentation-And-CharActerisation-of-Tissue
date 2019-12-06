#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 09:23:01 2019

@author: uqmtottr
"""
import os
from Model.config import src_path
from Preprocessing.data_prep_util import label_reorder, data_split, flip_traindata, label_distribution, reshuffle, rearrange, construct_exclude_vector

'''
Nipype directories
'''

dir_umc   = '/afm01/Q1/Q1219/data/Nipype_working_dir_20191014_UMC/output_dir/'
dir_mag   = '/afm01/Q1/Q1219/data/Nipype_working_dir_20191014_MAG/output_dir/'

data_path = os.path.join(src_path, 'preprocessed_data')
data_dest = os.path.join(src_path, 'data_for_network')

#Subjects to be excluded. Just an example

mag_right = ['000','002','005','006','007','010','012', '013', '014','015', '018','019','021','028','030','032','033']
mag_left = ['001','006','007','008','013','014','015','019','020','025','028']

# Construct exclusion vector
exclude_vector_mag = construct_exclude_vector(mag_right,mag_left)

# Reshuffle +                               exclusion (can include one for UMC and one for MAG, here just MAG, hence UMC is = [])
reshuffle(dir_umc, dir_mag, data_path, [[],exclude_vector_mag])

# Fix labelling, so both datasets have same labels
label_reorder(data_path)

# Split data into training and test and save respective adresses
addresses = data_split(data_path, data_dest)

# Boxplot of label distribution in the data (normalized) Shows and saves figure to src_path (where script is run from)
label_distribution(data_path)

# Flip the training data
flip_traindata(addresses[0], addresses[1], addresses[2], data_path)

# Rearrange data to desired input shape for the network (note, this could be combined with reshuffle to just do one shuffling step, 
                                                        #but time difference is negligble and it was implemented this way)
rearrange(data_path, data_dest)