#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 14:31:11 2019

@author: uqmtottr
"""

# Count number of voxels of each label in all images and calculate the average


from glob import glob
import numpy as np
import nibabel as nib


data_path = glob('/winmounts/uqmtottr/uq-research/DEEPSEACAT-Q1219/data/data_for_network/seg/*.nii.gz')


seg = []

l0 = []
l1 = []
l2 = []
l3 = []
l4 = []
l5 = []
l6 = []
l7 = []
l8 = []

res = [0, 0, 0, 0, 0, 0, 0, 0, 0]

for i in range(len(data_path)):
    seg.append(nib.load(data_path[i]))
    get_seg = seg[i].get_fdata()
    arr = np.array(get_seg)
    x = arr.copy()
    
    res[0] += len(x[x == 0])
    res[1] += len(x[x == 1])
    res[2] += len(x[x == 2])
    res[3] += len(x[x == 3])
    res[4] += len(x[x == 4])
    res[5] += len(x[x == 5])
    res[6] += len(x[x == 6])
    res[7] += len(x[x == 7])
    res[8] += len(x[x == 8])
    
    l0.append(len(x[x==0]))
    l1.append(len(x[x==1]))
    l2.append(len(x[x==2]))
    l3.append(len(x[x==3]))
    l4.append(len(x[x==4]))
    l5.append(len(x[x==5]))
    l6.append(len(x[x==6]))
    l7.append(len(x[x==7]))
    l8.append(len(x[x==8]))
    
avg_l0 = res[0]/200
avg_l1 = res[1]/200
avg_l2 = res[2]/200
avg_l3 = res[3]/200
avg_l4 = res[4]/200
avg_l5 = res[5]/200
avg_l6 = res[6]/200
avg_l7 = res[7]/200
avg_l8 = res[8]/200