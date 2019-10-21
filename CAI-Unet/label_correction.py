# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
from glob import glob
import numpy as np
import nibabel as nib

dir_path = '/winmounts/uqmtottr/uq-research/DEEPSEACAT-Q1219/data/data_for_network/'

### MAG segmentations ###
mag_seg_path = glob(dir_path+'seg/'+'*mag*'+'.nii.gz')
mag = []

for i in range(len(mag_seg_path)):
    mag.append(nib.load(mag_seg_path[i]))
    mag_get = mag[i].get_fdata()
    mag_arr = np.array(mag_get)
    x = mag_arr.copy()

    x[x == 5] = 0
    x[x == 6] = 0
    x[x == 7] = 0
    x[x == 10] = 0
    x[x == 11] = 0
    x[x == 12] = 0
    x[x == 13] = 0
    x[x == 17] = 0

    x[x == 3] = 5
    x[x == 4] = 6
    x[x == 2] = 4
    x[x == 8] = 2
    x[x == 1] = 3
    x[x == 9] = 1


    label_img = nib.Nifti1Image(x, affine=None)
    label_file = mag_seg_path[i][mag_seg_path[i].index('seg')+4:len(mag_seg_path[i])]   
    nib.save(label_img, os.path.join(dir_path+label_file))



### UMC segmentations ###
all_img = sorted(glob(dir_path+'seg/*'))
umc_seg_path = all_img[0:52]
umc = []

for j in range(len(umc_seg_path)):
    umc.append(nib.load(umc_seg_path[j]))
    umc_get = umc[j].get_fdata()
    umc_arr = np.array(umc_get)
    y = umc_arr.copy()
    
    y[y == 7] = 0
    y[y == 8] = 0
    
    lab_img = nib.Nifti1Image(y, affine=None)
    lab_file = umc_seg_path[j][umc_seg_path[j].index('seg')+4:len(umc_seg_path[j])]
    nib.save(lab_img, os.path.join(dir_path+lab_file))