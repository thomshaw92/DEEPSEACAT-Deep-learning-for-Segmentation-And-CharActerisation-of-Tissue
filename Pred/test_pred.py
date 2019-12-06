#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 16:41:50 2019

@author: uqdlund

Test to build the 3D unet model, can maybe serve as main later on.
"""
import sys
import os
import glob
sys.path.append(os.getcwd()) 

from utils import write_data_to_file

from predict_util import run_validation_cases



home_path = '/scratch/cai/DEEPSEACAT/data/20191120_leaky_ReLu_MAG_MPRAGE/'
if not os.path.exists(home_path):
    os.mkdir(home_path)

# Build config dictionary that might be needed later
config = dict()
config["image_shape"] = (176, 144, 128)     # This determines what shape the images will be cropped/resampled to.
config["labels"] = (0, 1, 2, 3, 4, 5, 6, 8)       # the label numbers on the input image
config["n_labels"] = len(config["labels"])  # Amount of labels
config["all_modalities"] = ["mprage"]     # Declare all available modalities
config["training_modalities"] = config["all_modalities"]    # change this if you want to only use some of the modalities
config["nb_channels"] = len(config["training_modalities"])  # Configures number of channels via number of modalities

config['data_path'] = '/scratch/cai/DEEPSEACAT/data/test_data_rearranged/'
config["data_file"] = '/scratch/cai/DEEPSEACAT/data/MAG_MPRAGE_test.hdf5'           # Typically hdf5 file 
config["model_file"] =      os.path.join(home_path, 'model.h5')          # If you have a model it will load model, if not it will save as this name
config["test_indices"] = list(range(5))
config["overwrite"] = False  # If True, will overwrite previous files. If False, will use previously written files.

# Fetches filenames
def fetch_training_data_files():
    training_data_files = list()
    for subject_dir in glob.glob(os.path.join(config['data_path'],'*')):#os.path.join(os.path.dirname(__file__), "data", "preprocessed", "*", "*")):
        subject_files = list()
        for modality in config["training_modalities"] + ["seg"]:
            subject_files.append(glob.glob(os.path.join(subject_dir, 'mag_'+modality+'*')))
        if subject_files[0]:
            training_data_files.append(tuple(subject_files))
        
    return training_data_files



def main(overwrite=False):
    # convert input images into an hdf5 file
    # if we want to overwrite files or it doen't exist create datafile
    if overwrite or not os.path.exists(config["data_file"]):
        print('fetching data files... \n')
        training_files = fetch_training_data_files()
        #print(training_files)
        print('writing data to file... \n')
        write_data_to_file(training_files, config["data_file"], image_shape=config["image_shape"])
        
    prediction_dir = os.path.join(home_path, 'test_prediction')
    print('running validation cases...')
                                                       ## NOTE ##
    run_validation_cases(validation_keys_file=config["test_indices"],
                         model_file=config['model_file'],
                         training_modalities=config["training_modalities"],
                         labels=config["labels"],
                         hdf5_file=config['data_file'],
                         output_label_map=True,
                         custom = True,
                         output_dir=prediction_dir)
    




if __name__ == "__main__":
    main(overwrite=config["overwrite"])
    
   
