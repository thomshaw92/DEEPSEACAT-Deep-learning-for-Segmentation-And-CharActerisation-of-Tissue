#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 16:41:50 2019

@author: uqdlund

Test to build the 3D unet model, can maybe serve as main later on.
"""
import sys
import os
sys.path.append(os.getcwd()) 
 
from Model.config import config, src_path
from Model.utils import fetch_training_data_files, write_data_to_file
from Pred.predict_util import run_validation_cases
from Pred.post_proc_util import Connected_components, Performance_metrics

# If you're not testing different stuff, you can just import all these configs from Model.config, model_file and labels are both loaded from config
config['test_data_path'] = os.path.join(src_path, 'test_data_for_network')
config["test_data_file"] = os.path.join(src_path, 'test.hdf5')
config["test_modalities"] = ["tse", "cd "]       

def main(overwrite=False):
    # convert input images into an hdf5 file
    # if we want to overwrite files or it doen't exist create datafile
    if overwrite or not os.path.exists(config["test_data_file"]):
        print('fetching data files... \n')
        training_files = fetch_training_data_files(config["test_data_path"], config["test_modalities"])
        print(training_files)
        
        var1 = len(config["test_modalities"])+1
        var2 = len(training_files)
        var3 = (var2 / var1)
        print(var1)
        print(var2)
        print(var3)
        test_indices = list(range(var2))
        
        
        print('writing data to file... \n')
        write_data_to_file(training_files, config["test_data_file"], image_shape=config["image_shape"])
        
    prediction_dir = os.path.join(src_path, 'test_prediction')
    print('running prediction cases...')
                                                       ## NOTE ##
    run_validation_cases(validation_keys_file=test_indices,
                         model_file=config['model_file'],
                         training_modalities=config["test_modalities"],
                         labels=config["labels"],
                         hdf5_file=config['test_data_file'],
                         output_label_map=True,
                         custom = True,
                         output_dir=prediction_dir)
    
    print('Calculating performance...')
    # runs connected component analysis on predicted images and returns path to new images
    CC_path = Connected_components(prediction_dir, config["labels"])
      
    #### Performance Metrics ####
    # for each subject:
    # vol contains [predicted_volumes, true_volumes, volume_deviations]
    # dice contains dice for the whole segmentation
    # dice_label and SBD contains dice and SBD for each label
    dice, dice_label, SBD, vol = Performance_metrics(CC_path, config["labels"])
    


if __name__ == "__main__":
    main(overwrite=config["overwrite"])
    
   
