#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 13:02:07 2019

@author: uqmtottr
"""

# Count number of voxels of each label0-6+8 in all images and calculate the average


from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import scipy.stats as stat

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



# Normalization of volume data
# Manual segmentations
sub_sum = []
norm1 = []
norm2 = []
norm3 = []
norm4 = []
norm5 = []
norm6 = []
norm8 = []

for i in range(11):
    sub_sum = l1[i]+l2[i]+l3[i]+l4[i]+l5[i]+l6[i]+l8[i]
    norm1.append(l1[i]/sub_sum)
    norm2.append(l2[i]/sub_sum)
    norm3.append(l3[i]/sub_sum)
    norm4.append(l4[i]/sub_sum)
    norm5.append(l5[i]/sub_sum)
    norm6.append(l6[i]/sub_sum)
    norm8.append(l8[i]/sub_sum)

norm1_avg = np.mean(norm1)
norm2_avg = np.mean(norm2)
norm3_avg = np.mean(norm3)
norm4_avg = np.mean(norm4)
norm5_avg = np.mean(norm5)
norm6_avg = np.mean(norm6)
norm8_avg = np.mean(norm8)

norm1_std = np.std(norm1)
norm2_std = np.std(norm2)
norm3_std = np.std(norm3)
norm4_std = np.std(norm4)
norm5_std = np.std(norm5)
norm6_std = np.std(norm6)
norm8_std = np.std(norm8)

norm_manual = [norm1, norm2, norm3, norm4, norm5, norm6, norm8]
norm_manual = np.reshape(norm_manual,[7,11])
norm_manual = norm_manual.transpose()
np.save('/clusterdata/uqmtottr/DEEPSEACAT-Deep-learning-for-Segmentation-And-CharActerisation-of-Tissue/CAI-Unet/norm_manual', norm_manual)

# Predicted segmentations
sub_sum_pred = []
norm1_pred = []
norm2_pred = []
norm3_pred = []
norm4_pred = []
norm5_pred = []
norm6_pred = []
norm8_pred = []

for i in range(11):
    sub_sum_pred = p1[i]+p2[i]+p3[i]+p4[i]+p5[i]+p6[i]+p8[i]
    norm1_pred.append(p1[i]/sub_sum_pred)
    norm2_pred.append(p2[i]/sub_sum_pred)
    norm3_pred.append(p3[i]/sub_sum_pred)
    norm4_pred.append(p4[i]/sub_sum_pred)
    norm5_pred.append(p5[i]/sub_sum_pred)
    norm6_pred.append(p6[i]/sub_sum_pred)
    norm8_pred.append(p8[i]/sub_sum_pred)

norm1_avg_pred = np.mean(norm1_pred)
norm2_avg_pred = np.mean(norm2_pred)
norm3_avg_pred = np.mean(norm3_pred)
norm4_avg_pred = np.mean(norm4_pred)
norm5_avg_pred = np.mean(norm5_pred)
norm6_avg_pred = np.mean(norm6_pred)
norm8_avg_pred = np.mean(norm8_pred)

norm1_std_pred = np.std(norm1_pred)
norm2_std_pred = np.std(norm2_pred)
norm3_std_pred = np.std(norm3_pred)
norm4_std_pred = np.std(norm4_pred)
norm5_std_pred = np.std(norm5_pred)
norm6_std_pred = np.std(norm6_pred)
norm8_std_pred = np.std(norm8_pred)

norm_pred = [norm1_pred, norm2_pred, norm3_pred, norm4_pred, norm5_pred, norm6_pred, norm8_pred]
norm_pred = np.reshape(norm_pred,[7,11])
norm_pred = norm_pred.transpose()
np.save('/clusterdata/uqmtottr/DEEPSEACAT-Deep-learning-for-Segmentation-And-CharActerisation-of-Tissue/CAI-Unet/norm_pred', norm_pred)


### T-test calculation ###
ttest = stat.ttest_ind(norm_manual, norm_pred)

### BOXPLOT ###
box_plot_data=[norm_pred[:,0], norm_pred[:,1], norm_pred[:,2], norm_pred[:,3], norm_pred[:,4], norm_pred[:,5], norm_pred[:,6], np.mean(norm_pred,axis=0)]
box_plot_data1=[norm_manual[:,0], norm_manual[:,1], norm_manual[:,2], norm_manual[:,3], norm_manual[:,4], norm_manual[:,5], norm_manual[:,6], np.mean(norm_manual,axis=0)]
box=plt.boxplot(box_plot_data, labels=['ERC', 'Sub','CA1','CA2$^a$', 'DG', 'CA3', 'Tail', 'Mean'], patch_artist=True, showfliers=False, positions=np.array(range(len(box_plot_data)))*2.0-0.4, widths = 0.5)
box1=plt.boxplot(box_plot_data1, labels=['', '','','', '', '', '', ''], patch_artist=True, showfliers=False, positions=np.array(range(len(box_plot_data1)))*2.0+0.2, widths = 0.5)
plt.title('Volume for Proposed and Manual Method')
plt.ylabel('Normalized volume')
colors = ['#984ea3']*8
colors1 = ['#df65b0']*8
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
for median in box['medians']:
    median.set(color='black')
for patch, color in zip(box1['boxes'], colors1):
    patch.set_facecolor(color)
for median in box1['medians']:
    median.set(color='black')
plt.legend([box["boxes"][0], box1["boxes"][0]], ['Proposed method', 'Manual method'], loc='upper right')
plt.savefig('/scratch/cai/DEEPSEACAT/data/Boxplot_vol.eps', format = 'eps')
plt.show()
