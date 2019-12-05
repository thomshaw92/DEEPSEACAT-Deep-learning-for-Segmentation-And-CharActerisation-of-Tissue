#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 09:23:01 2019

@author: uqmtottr
"""

from data_prep_util import label_reorder, data_split, flip_traindata, label_distribution, reshuffle, rearrange, construct_exclude_vector

'''
Nipype directories
'''

dir_umc   = '/afm01/Q1/Q1219/data/Nipype_working_dir_20191014_UMC/output_dir/'
dir_mag   = '/afm01/Q1/Q1219/data/Nipype_working_dir_20191014_MAG/output_dir/'

data_path = '/scratch/cai/DEEPSEACAT/data/preprocessed_data/'
data_dest = '/scratch/cai/DEEPSEACAT/data/data_for_network/'

'''
Subjects to be excluded
'''
mag_right = ['000','002','005','006','007','010','012', '013', '014','015', '018','019','021','028','030','032','033']
mag_left = ['001','006','007','008','013','014','015','019','020','025','028']

exclude_vector_mag = construct_exclude_vector(mag_right,mag_left)

reshuffle(dir_umc, dir_mag, data_path, [[],exclude_vector_mag])

label_reorder(data_path)

addresses = data_split(data_path, data_dest)

label_distribution(data_path)

flip_traindata(addresses[0], addresses[1], addresses[2], data_path)

rearrange(data_path, data_dest)