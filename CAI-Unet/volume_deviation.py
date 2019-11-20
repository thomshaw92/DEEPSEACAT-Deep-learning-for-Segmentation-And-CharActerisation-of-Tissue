#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 13:02:07 2019

@author: uqmtottr
"""

# Count number of voxels of each label0-6+8 in all images and calculate the average


from glob import glob
import numpy as np
import nibabel as nib


data_path = glob('/scratch/cai/DEEPSEACAT/data/20191107_leaky_ReLu/test_prediction/time_test/validation*/truth.nii.gz')
pred_path = glob('/scratch/cai/DEEPSEACAT/data/20191107_leaky_ReLu/test_prediction/time_test/CC/*/*.nii.gz')

### Volume calculation true segmentations of the test data ###
seg = []
l1 = []
l2 = []
l3 = []
l4 = []
l5 = []
l6 = []
l8 = []

#Find the total number of voxels pr. label for all 11 subejcts
for i in range(len(data_path)):
    seg.append(nib.load(data_path[i]))
    get_seg = seg[i].get_fdata()
    arr = np.array(get_seg)
    x = arr.copy()
    
    l1.append(len(x[x==1]))
    l2.append(len(x[x==2]))
    l3.append(len(x[x==3]))
    l4.append(len(x[x==4]))
    l5.append(len(x[x==5]))
    l6.append(len(x[x==6]))
    l8.append(len(x[x==8]))

#Find the average number of voxels pr. subject   
avg_l1 = sum(l1)/11
avg_l2 = sum(l2)/11
avg_l3 = sum(l3)/11
avg_l4 = sum(l4)/11
avg_l5 = sum(l5)/11
avg_l6 = sum(l6)/11
avg_l8 = sum(l8)/11

#Calculate volume instead of number of voxel
l1_volume = avg_l1*0.35**3
l2_volume = avg_l2*0.35**3
l3_volume = avg_l3*0.35**3
l4_volume = avg_l4*0.35**3
l5_volume = avg_l5*0.35**3
l6_volume = avg_l6*0.35**3
l8_volume = avg_l8*0.35**3

#Calculate std for each label
l1_std = np.std(l1)*(0.35**3)
l2_std = np.std(l2)*(0.35**3)
l3_std = np.std(l3)*(0.35**3)
l4_std = np.std(l4)*(0.35**3)
l5_std = np.std(l5)*(0.35**3)
l6_std = np.std(l6)*(0.35**3)
l8_std = np.std(l8)*(0.35**3)
    
### Volume calculation predicted segmentations of the test data ###   
pred = []
p1 = []
p2 = []
p3 = []
p4 = []
p5 = []
p6 = []
p8 = []

for j in range(len(pred_path)):
    pred.append(nib.load(pred_path[j]))
    get_pred = pred[j].get_fdata()
    arr = np.array(get_pred)
    y = arr.copy()
    
    p1.append(len(y[y==1]))
    p2.append(len(y[y==2]))
    p3.append(len(y[y==3]))
    p4.append(len(y[y==4]))
    p5.append(len(y[y==5]))
    p6.append(len(y[y==6]))
    p8.append(len(y[y==8]))
    
avg_p1 = sum(p1)/11
avg_p2 = sum(p2)/11
avg_p3 = sum(p3)/11
avg_p4 = sum(p4)/11
avg_p5 = sum(p5)/11
avg_p6 = sum(p6)/11
avg_p8 = sum(p8)/11

#Calculate volume instead of number of voxel
p1_volume = avg_p1*0.35**3
p2_volume = avg_p2*0.35**3
p3_volume = avg_p3*0.35**3
p4_volume = avg_p4*0.35**3
p5_volume = avg_p5*0.35**3
p6_volume = avg_p6*0.35**3
p8_volume = avg_p8*0.35**3

p1_std = np.std(p1)*(0.35**3)
p2_std = np.std(p2)*(0.35**3)
p3_std = np.std(p3)*(0.35**3)
p4_std = np.std(p4)*(0.35**3)
p5_std = np.std(p5)*(0.35**3)
p6_std = np.std(p6)*(0.35**3)
p8_std = np.std(p8)*(0.35**3)

# Calculation of the volume deviation of each label between the true segmentation and the predicted segmentation of the 11 swubjects in the testset
vol_dev1 = p1_volume - l1_volume
vol_dev2 = p2_volume - l2_volume
vol_dev3 = p3_volume - l3_volume
vol_dev4 = p4_volume - l4_volume
vol_dev5 = p5_volume - l5_volume
vol_dev6 = p6_volume - l6_volume
vol_dev8 = p8_volume - l8_volume

dev_percentage1 = (vol_dev1/l1_volume)*100
dev_percentage2 = (vol_dev2/l2_volume)*100
dev_percentage3 = (vol_dev3/l3_volume)*100
dev_percentage4 = (vol_dev4/l4_volume)*100
dev_percentage5 = (vol_dev5/l5_volume)*100
dev_percentage6 = (vol_dev6/l6_volume)*100
dev_percentage8 = (vol_dev8/l8_volume)*100
