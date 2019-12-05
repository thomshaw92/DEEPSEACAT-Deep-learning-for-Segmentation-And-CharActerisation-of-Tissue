#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 09:23:01 2019

@author: uqmtottr
"""
import os
from glob import glob
import random
import shutil
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


def construct_exclude_vector(right_exclude_vector, left_exclude_vector):
# Constructs exclusion vector of a left and right sided array
    exclude_vector = []
    exclude_vector.append(['train'+idx for idx in right_exclude_vector])
    exclude_vector.append(['train'+idx for idx in left_exclude_vector])
    
    return exclude_vector
    

def reshuffle(input_dir1, input_dir2, dir_dest, exclude_vector = [[],[]]):
    # Reshuffles data into a folder structure used for remaining preprocessing
    umc = ['UMC_normalized_TSE','UMC_normalized_MPRAGE', 'UMC_reslice_labels_SEG']
    
    mag = ['MAG_normalized_TSE','MAG_normalized_MPRAGE','MAG_reslice_labels_SEG']
    
    # Sort lists
    umc_lists = []
    for mode in umc:
        umc_lists.append(sorted(glob(os.path.join(input_dir1,mode,'*','*','*'))))
    
    mag_lists = []
    for mode in mag:
        mag_lists.append(sorted(glob(os.path.join(input_dir1,mode,'*','*','*'))))
    
    if not os.path.exists(dir_dest):
        os.mkdir(dir_dest)
        os.mkdir(dir_dest+'tse/')
        os.mkdir(dir_dest+'mprage/')
        os.mkdir(dir_dest+'seg/')
    
    # Can be made neater with additional for loop and better variable naming, but eh, it's decently fast
    if not exclude_vector[0]:
        for i in range(len(umc_lists[0])):
            if 'left' in umc_lists[0][i]:
                shutil.copy(umc_lists[0][i], dir_dest+'tse/'+umc_lists[0][i][umc_lists[0][i].index('train')+5:umc_lists[0][i].index('train')+8]+'_tse'+'_left.nii.gz')
                shutil.copy(umc_lists[1][i], dir_dest+'mprage/'+umc_lists[1][i][umc_lists[1][i].index('train')+5:umc_lists[1][i].index('train')+8]+'_mprage'+'_left.nii.gz')
                shutil.copy(umc_lists[2][i], dir_dest+'seg/'+umc_lists[2][i][umc_lists[2][i].index('train')+5:umc_lists[2][i].index('train')+8]+'_seg'+'_left.nii.gz')
            elif 'right' in umc_lists[0][i]:
                shutil.copy(umc_lists[0][i], dir_dest+'tse/'+umc_lists[0][i][umc_lists[0][i].index('train')+5:umc_lists[0][i].index('train')+8]+'_tse'+'_right.nii.gz')
                shutil.copy(umc_lists[1][i], dir_dest+'mprage/'+umc_lists[1][i][umc_lists[1][i].index('train')+5:umc_lists[1][i].index('train')+8]+'_mprage'+'_right.nii.gz')
                shutil.copy(umc_lists[2][i], dir_dest+'seg/'+umc_lists[2][i][umc_lists[2][i].index('train')+5:umc_lists[2][i].index('train')+8]+'_seg'+'_right.nii.gz')
    else:
        for i in range(len(umc_lists[0])):
            if 'left' in umc_lists[0][i] and not any([temp in umc_lists[0][i] for temp in exclude_vector[0][1]]):
                shutil.copy(umc_lists[0][i], dir_dest+'tse/'+umc_lists[0][i][umc_lists[0][i].index('train')+5:umc_lists[0][i].index('train')+8]+'_tse'+'_left.nii.gz')
                shutil.copy(umc_lists[1][i], dir_dest+'mprage/'+umc_lists[1][i][umc_lists[1][i].index('train')+5:umc_lists[1][i].index('train')+8]+'_mprage'+'_left.nii.gz')
                shutil.copy(umc_lists[2][i], dir_dest+'seg/'+umc_lists[2][i][umc_lists[2][i].index('train')+5:umc_lists[2][i].index('train')+8]+'_seg'+'_left.nii.gz')
            elif 'right' in umc_lists[0][i] and not any([temp in umc_lists[0][i] for temp in exclude_vector[0][0]]):
                shutil.copy(umc_lists[0][i], dir_dest+'tse/'+umc_lists[0][i][umc_lists[0][i].index('train')+5:umc_lists[0][i].index('train')+8]+'_tse'+'_right.nii.gz')
                shutil.copy(umc_lists[1][i], dir_dest+'mprage/'+umc_lists[1][i][umc_lists[1][i].index('train')+5:umc_lists[1][i].index('train')+8]+'_mprage'+'_right.nii.gz')
                shutil.copy(umc_lists[2][i], dir_dest+'seg/'+umc_lists[2][i][umc_lists[2][i].index('train')+5:umc_lists[2][i].index('train')+8]+'_seg'+'_right.nii.gz')
        
        
    if not exclude_vector[1]:
        for i in range(len(mag_lists[0])):
            if 'left' in mag_lists[0][i]:
                shutil.copy(mag_lists[0][i], dir_dest+'tse/0'+str(int(mag_lists[0][i][mag_lists[0][i].index('train')+5:mag_lists[0][i].index('train')+8])+int(len(umc_lists[0])/2))+'_mag_tse'+'_left.nii.gz')
                shutil.copy(mag_lists[1][i], dir_dest+'mprage/0'+str(int(mag_lists[1][i][mag_lists[1][i].index('train')+5:mag_lists[1][i].index('train')+8])+int(len(umc_lists[0])/2))+'_mag_mprage'+'_left.nii.gz')
                shutil.copy(mag_lists[2][i], dir_dest+'seg/0'+str(int(mag_lists[2][i][mag_lists[2][i].index('train')+5:mag_lists[2][i].index('train')+8])+int(len(umc_lists[0])/2))+'_mag_seg'+'_left.nii.gz')
            if 'right' in mag_lists[0][i]:#right_vector_orig in mag_lists[0][i]:
                shutil.copy(mag_lists[0][i], dir_dest+'tse/0'+str(int(mag_lists[0][i][mag_lists[0][i].index('train')+5:mag_lists[0][i].index('train')+8])+int(len(umc_lists[0])/2))+'_mag_tse'+'_right.nii.gz')
                shutil.copy(mag_lists[1][i], dir_dest+'mprage/0'+str(int(mag_lists[1][i][mag_lists[1][i].index('train')+5:mag_lists[1][i].index('train')+8])+int(len(umc_lists[0])/2))+'_mag_mprage'+'_right.nii.gz')
                shutil.copy(mag_lists[2][i], dir_dest+'seg/0'+str(int(mag_lists[2][i][mag_lists[2][i].index('train')+5:mag_lists[2][i].index('train')+8])+int(len(umc_lists[0])/2))+'_mag_seg'+'_right.nii.gz')
    else:
        for i in range(len(mag_lists[0])):
            if 'left' in mag_lists[0][i] and not any([temp in mag_lists[0][i] for temp in exclude_vector[1][1]]):
                shutil.copy(mag_lists[0][i], dir_dest+'tse/0'+str(int(mag_lists[0][i][mag_lists[0][i].index('train')+5:mag_lists[0][i].index('train')+8])+int(len(umc_lists[0])/2))+'_mag_tse'+'_left.nii.gz')
                shutil.copy(mag_lists[1][i], dir_dest+'mprage/0'+str(int(mag_lists[1][i][mag_lists[1][i].index('train')+5:mag_lists[1][i].index('train')+8])+int(len(umc_lists[0])/2))+'_mag_mprage'+'_left.nii.gz')
                shutil.copy(mag_lists[2][i], dir_dest+'seg/0'+str(int(mag_lists[2][i][mag_lists[2][i].index('train')+5:mag_lists[2][i].index('train')+8])+int(len(umc_lists[0])/2))+'_mag_seg'+'_left.nii.gz')
            if 'right' in mag_lists[0][i] and not any([temp in mag_lists[0][i] for temp in exclude_vector[1][0]]):
                shutil.copy(mag_lists[0][i], dir_dest+'tse/0'+str(int(mag_lists[0][i][mag_lists[0][i].index('train')+5:mag_lists[0][i].index('train')+8])+int(len(umc_lists[0])/2))+'_mag_tse'+'_right.nii.gz')
                shutil.copy(mag_lists[1][i], dir_dest+'mprage/0'+str(int(mag_lists[1][i][mag_lists[1][i].index('train')+5:mag_lists[1][i].index('train')+8])+int(len(umc_lists[0])/2))+'_mag_mprage'+'_right.nii.gz')
                shutil.copy(mag_lists[2][i], dir_dest+'seg/0'+str(int(mag_lists[2][i][mag_lists[2][i].index('train')+5:mag_lists[2][i].index('train')+8])+int(len(umc_lists[0])/2))+'_mag_seg'+'_right.nii.gz')
 

def label_reorder(data_path):
    '''
    Run only once, as it will overwrite the wrong labels otherwise || can be made safe by checking if all mag_array[mag_arr == to_zero] return empty indices
    Label correction of the MAG data set
    As data is collected from two different data sets (UMC and MAG) the labels are numberede/positionated differently.
    The numbers are based on the ones given in the UMC dataset as listed below. 
    The labels in MAG that are not listed below will be set to zero to make them part of the background.
    
    UMC label numbers, which the MAG will be corrected according to.
    0 = Background / Clear label
    1 = ERC
    2 = SUB
    3 = CA1
    4 = CA2
    5 = DG
    6 = CA3
    8 = Tail
    '''
    mag_seg = glob(data_path+'seg/'+'*mag*'+'.nii.gz')

    # target vectors
    to_zero = [6,7,10,11,12,17]
    # necessary order for forloop not to overwrite existing labels
    to_change = [13,4,2,8,5,3,1,9]
    change_target = [7,6,4,2,8,5,3,1]
    
    for i in range(len(mag_seg)):
        mag_arr = (nib.load(mag_seg[i])).get_fdata()
        
        # set unnecessary labels to zero
        for label in to_zero:
            mag_arr[mag_arr == label] = 0
        
        # Change to UMC labels
        for j in range(len(to_change)):
            mag_arr[mag_arr == to_change[j]] = change_target[j]

        # overwrite old segmentation with new, through Nibabel
        label_img = nib.Nifti1Image(mag_arr, affine=None)
        label_file = mag_seg[i][mag_seg[i].index('seg')+4:len(mag_seg[i])]   
        nib.save(label_img, data_path+'seg/'+label_file) # second argument could just be mag_seg[i] I believe


def data_split(data_path,data_dest):
    '''
    Run only once, as the test set will be overwritten otherwise.
    Split data into a test (10%) and train (90%) set (the train set will be split into train and validation later on)
    The data is extracted and saved in new lists test/train_.._addrs
    '''
    
    # Define paths
    tse = sorted(glob(data_path + 'tse/*.nii.gz'))
    mprage = sorted(glob(data_path + 'mprage/*.nii.gz'))
    seg = sorted(glob(data_path + 'seg/*.nii.gz'))
    
    test = os.path.join(data_dest, 'test/')
    test_tse = os.path.join(data_dest, 'test/tse/')
    test_mprage = os.path.join(data_dest, 'test/mprage/')
    test_seg = os.path.join(data_dest, 'test/seg/')

    # make dirs if not done already
    if not os.path.exists(data_dest):
        os.mkdir(data_dest)
    if not os.path.exists(test):
        os.mkdir(test)
        for type in [test_tse,test_mprage,test_seg]:
            os.mkdir(type)
    
    n_sub = list(range(len(tse))) #The lenght of tse, mprage and seg paths are the same, so here we just use tse
    testset_sub = sorted(random.sample(n_sub, round(len(tse)*0.1))) # extract 10 % of all data for test

    test_tse_addrs = []
    test_mprage_addrs = []
    test_seg_addrs = []
    train_tse_addrs = []
    train_mprage_addrs = []
    train_seg_addrs = []

    # Separate the extracted test samples to respective types
    for i in range(len(tse)):
        if i in testset_sub:
            test_tse_addrs.append(tse[i])
            test_mprage_addrs.append(mprage[i])
            test_seg_addrs.append(seg[i])
        else:
            train_tse_addrs.append(tse[i])
            train_mprage_addrs.append(mprage[i])
            train_seg_addrs.append(seg[i])
            
    for k in range(len(test_tse_addrs)):
        shutil.move(test_tse_addrs[k], test_tse)
        shutil.move(test_mprage_addrs[k], test_mprage)
        shutil.move(test_seg_addrs[k], test_seg)
    
    return [train_tse_addrs, train_mprage_addrs, train_seg_addrs]


def label_distribution(data_path):        
    '''
    Calculation of the volume deviation of each label based on the manual segmentations
    The volume is calculated in mm based on the resolution set in the nipype preprocessing (voxel resolution = 0.35 mm^3 iso)
    Each subfield is divided by the total hippocampal volume of all subjects to get a proportion (% of hippocampus volume)
    A subplot is generated to visualize the label proportion of the hippocampal volume.
    '''  
    segmentations = glob(data_path + 'seg/*.nii.gz')
    seg = []
    lab = []
    
    for subject in range(len(segmentations)):
        seg.append(nib.load(segmentations[subject]))
        get_seg = seg[subject].get_fdata()
        arr = np.array(get_seg)
        temp=[]
        for label in (1,2,3,4,5,6,8):
            temp.append(len(arr[arr==label]))
        lab.append(temp)
    
    lab = np.asarray(lab)
    lab_vol = lab*0.35**3 # to mm resolution
    
    hippo_volumes = []
    lab_proportions = []
    for sub in range(len(segmentations)):
        hippo_volumes.append(sum(lab_vol[sub,:]))
        lab_proportions.append(lab_vol[sub,:]/hippo_volumes[sub]) # relative/proportional size
    lab_proportions = np.asarray(lab_proportions)
    
    box_plot_data=[lab_proportions[:,0], lab_proportions[:,1], lab_proportions[:,2], lab_proportions[:,3], lab_proportions[:,4], lab_proportions[:,5], lab_proportions[:,6]]
    box=plt.boxplot(box_plot_data, widths = 0.7, labels=['ERC','Sub','CA1','CA2', 'DG', 'CA3', 'Tail'], patch_artist=True, showfliers=False)
    plt.title('Distribution of Labels')
    plt.ylabel('% of Hippocampal Volume')
    #plt.yticks(np.arange(0, 2250, step=250))
    colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d"]
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    for median in box['medians']:
        median.set(color='black')
    plt.savefig('LabelDistributionAbstract.eps', format = 'eps')
    plt.show()  


def flip_traindata(train_tse_addrs, train_mprage_addrs, train_seg_addrs, data_path):
    '''
    Left-right flip (in the z-axis) data in the training list as a data augmentation
    '''
    addresses =[train_tse_addrs, train_mprage_addrs, train_seg_addrs]
    for j in range(len(addresses[0])):
        for address in addresses:
            img_arr = (nib.load(address[j])).get_fdata()
            img_flip = img_arr[:, :, ::-1]
            nifti_img = nib.Nifti1Image(img_flip, affine = None)
            
            if 'tse' in address[j]:
                filename_flipped_tse = 'flipped_' + train_tse_addrs[j][train_tse_addrs[j].index('tse')+4:len(train_tse_addrs[j])]
                nib.save(nifti_img, os.path.join(data_path + 'tse/' + filename_flipped_tse))
            if 'mprage' in address[j]:
                filename_flipped_mprage = 'flipped_' + train_mprage_addrs[j][train_mprage_addrs[j].index('mprage')+7:len(train_mprage_addrs[j])]
                nib.save(nifti_img, os.path.join(data_path + 'mprage/' + filename_flipped_mprage))
            if 'seg' in address[j]:
                filename_flipped_seg = 'flipped_' + train_seg_addrs[j][train_seg_addrs[j].index('seg')+4:len(train_seg_addrs[j])]
                nib.save(nifti_img, os.path.join(data_path + 'seg/' + filename_flipped_seg))
               
                
def rearrange(src_path, dest_path, targets =['tse','mprage','seg'], flipped = True):
    '''
    Final rearrange of data, to a form the generator in the network expects
    splits data into subjects with respective targets in each subject folder (default is [tse,mprageg, seg])
    If flipped = True, then it will look for flipped images as well and name them accordingly, default is True
    '''
    if not os.path.exists(dest_path):
        os.mkdir(dest_path)

    for im_type in targets:
        array = sorted(os.listdir(os.path.join(src_path,im_type)))
        list_ = []
        
        for subject in array:
            list_.append(subject[0:3])
    
        for subject in array:
            if 'right' in subject and not 'flipped' in subject:
                if not os.path.exists(os.path.join(dest_path,subject[0:3]+'_right')):
                    os.mkdir(os.path.join(dest_path,subject[0:3]+'_right'))
            elif 'right' in subject and 'flipped' in subject and flipped:
                if not os.path.exists(os.path.join(dest_path,subject[8:11]+'_right_flipped')):
                    os.mkdir(os.path.join(dest_path,subject[8:11]+'_right_flipped'))
            elif 'left' in subject and 'flipped' in subject and flipped:
                if not os.path.exists(os.path.join(dest_path,subject[8:11]+'_left_flipped')):
                    os.mkdir(os.path.join(dest_path,subject[8:11]+'_left_flipped'))    
            elif not 'flipped' in subject:
                if not os.path.exists(os.path.join(dest_path,subject[0:3]+'_left')):
                    os.mkdir(os.path.join(dest_path,subject[0:3]+'_left'))
    
        for subject in array:
            for i in range(len(array)):
                if subject[0:3] == list_[i] and 'right' in subject and not 'flipped' in subject:
                    shutil.copy(os.path.join(src_path,im_type,subject),os.path.join(dest_path,list_[i]+'_right',subject[4:len(subject)]))
                    break
                elif subject[0:3] == list_[i] and 'left' in subject and not 'flipped' in subject:
                    if subject[0:3] == list_[i]:
                        shutil.copy(os.path.join(src_path,im_type,subject),os.path.join(dest_path, list_[i]+'_left',subject[4:len(subject)]))
                        break
                elif subject[8:11] == list_[i] and 'right' in subject and 'flipped' in subject and flipped:
                    if subject[8:11] == list_[i]:
                        shutil.copy(os.path.join(src_path,im_type,subject),os.path.join(dest_path, list_[i]+'_right_flipped',subject[0:8] + subject[12:len(subject)]))
                        break
                elif flipped:
                    if subject[8:11] == list_[i]:
                        shutil.copy(os.path.join(src_path,im_type,subject),os.path.join(dest_path, list_[i]+'_left_flipped',subject[0:8] + subject[12:len(subject)]))
                        break