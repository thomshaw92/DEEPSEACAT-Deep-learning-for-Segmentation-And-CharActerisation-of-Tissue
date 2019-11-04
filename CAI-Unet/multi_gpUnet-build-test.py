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

import tensorflow as tf
from keras.utils import plot_model#, multi_gpu_model
from keras.models import load_model
from keras.layers import Lambda, concatenate
from keras import Model
from keras.optimizers import Adam


from model import unet_model_3d
from Metrics import dice_coefficient_loss, get_label_dice_coefficient_function, dice_coefficient
from utils import write_data_to_file, open_data_file, get_callbacks
from generator import get_training_and_validation_generators
from predict_util import run_validation_cases


def multi_gpu_model(model, gpus):
  if isinstance(gpus, (list, tuple)):
    num_gpus = len(gpus)
    target_gpu_ids = gpus
  else:
    num_gpus = gpus
    target_gpu_ids = range(num_gpus)

  def get_slice(data, i, parts):
    shape = tf.shape(data)
    batch_size = shape[:1]
    input_shape = shape[1:]
    step = batch_size // parts
    if i == num_gpus - 1:
      size = batch_size - step * i
    else:
      size = step
    size = tf.concat([size, input_shape], axis=0)
    stride = tf.concat([step, input_shape * 0], axis=0)
    start = stride * i
    return tf.slice(data, start, size)

  all_outputs = []
  for i in range(len(model.outputs)):
    all_outputs.append([])

  # Place a copy of the model on each GPU,
  # each getting a slice of the inputs.
  for i, gpu_id in enumerate(target_gpu_ids):
    with tf.device('/gpu:%d' % gpu_id):
      with tf.name_scope('replica_%d' % gpu_id):
        inputs = []
        # Retrieve a slice of the input.
        for x in model.inputs:
          input_shape = tuple(x.get_shape().as_list())[1:]
          slice_i = Lambda(get_slice,
                           output_shape=input_shape,
                           arguments={'i': i,
                                      'parts': num_gpus})(x)
          inputs.append(slice_i)

        # Apply model on slice
        # (creating a model replica on the target device).
        outputs = model(inputs)
        if not isinstance(outputs, list):
          outputs = [outputs]

        # Save the outputs for merging back together later.
        for o in range(len(outputs)):
          all_outputs[o].append(outputs[o])

  # Merge outputs on CPU.
  with tf.device('/cpu:0'):
    merged = []
    for name, outputs in zip(model.output_names, all_outputs):
      merged.append(concatenate(outputs,
                                axis=0, name=name))
    return Model(model.inputs, merged)



home_path = '/scratch/cai/DEEPSEACAT/data/20191023_multi_gpu_whole/'
if not os.path.exists(home_path):
    os.mkdir(home_path)

# Build config dictionary that might be needed later
config = dict()
config["depth"] = 4
config["strided_conv_size"] = (2, 2, 2)     # Size for the strided convolutional operations
config["image_shape"] = (176, 144, 128)     # This determines what shape the images will be cropped/resampled to.
config["patch_shape"] = None                # None = Train on the whole image, switch to specific dimensions if patch extraction is needed
config["labels"] = (1, 2, 3, 4, 5, 6, 7 ,8)       # the label numbers on the input image, should the 0 label be included??
config["n_labels"] = len(config["labels"])  # Amount of labels
config["all_modalities"] = ["tse", "mprage"]     # Declare all available modalities
config["training_modalities"] = config["all_modalities"]    # change this if you want to only use some of the modalities
config["nb_channels"] = len(config["training_modalities"])  # Configures number of channels via number of modalities
if "patch_shape" in config and config["patch_shape"] is not None:       # Determine input shape, based on patch or not
    config["input_shape"] = tuple([config["nb_channels"]] + list(config["patch_shape"]))
else:
    config["input_shape"] = tuple([config["nb_channels"]] + list(config["image_shape"]))
config["dilation_block"]    = False
config["n_dil_block"]       = 1             # Must be at least 1 lower than depth
config["residual"]          = True
config["dense"]             = False
config["truth_channel"] = config["nb_channels"]
config["deconvolution"] = True          # if False, will use upsampling instead of deconvolution
config["n_base_filters"] = 16   # Tested at 32, no OOM
config["batch_size"] = 18       # Tested at 32
config["validation_batch_size"] = 100
config["n_epochs"] = 10                  # cutoff the training after this many epochs
config["patience"] = 5                  # learning rate will be reduced after this many epochs if the validation loss is not improving
config["early_stop"] = 15               # training will be stopped after this many epochs without the validation loss improving
config["initial_learning_rate"] = 1e-5
config["learning_rate_drop"] = 0.5                                  # factor by which the learning rate will be reduced
config["learning_rate_epochs"] = None   # Number of epochs after which the learning rate will drop.
config["validation_split"] = 0.9                                    # portion of the data that will be used for training
#config["flip"] = False                                              # augments the data by randomly flipping an axis during
#config["permute"] = False               # data shape must be a cube. Augments the data by permuting in various directions
#config["distort"] = None                                            # switch to None if you want no distortion
#config["augment"] = config["flip"] or config["distort"]
config["validation_patch_overlap"] = 0                              # if > 0, during training, validation patches will be overlapping
config["training_patch_start_offset"] = (16, 16, 16)                # randomly offset the first patch index by up to this offset
config["skip_blank"] = True                                         # if True, then patches without any target will be skipped

config['logging_file'] = os.path.join(home_path, 'training.log')
config['data_path'] = '/scratch/cai/DEEPSEACAT/data/data_config_flipped/'
config["data_file"] =       os.path.join(home_path, 'train_val.hdf5')    # Typically hdf5 file 'train_val_flipped.hdf5' and 'train_val_100.hdf5'
config["model_file"] =      os.path.join(home_path, 'model.h5')          # If you have a model it will load model, if not it will save as this name
config["training_file"] =   os.path.join(home_path, 'training_ids.pkl')  # Same
config["validation_file"] = os.path.join(home_path, 'validation_ids.pkl')
config["overwrite"] = True  # If True, will overwrite previous files. If False, will use previously written files.

# Some code to fetch data #


# Fetches filenames
def fetch_training_data_files():
    training_data_files = list()
    for subject_dir in glob.glob(os.path.join(config['data_path'],'*')):#os.path.join(os.path.dirname(__file__), "data", "preprocessed", "*", "*")):
        subject_files = list()
        for modality in config["training_modalities"] + ["*seg*"]:
            subject_files.append(glob.glob(os.path.join(subject_dir, '*'+modality+'*')))
        training_data_files.append(tuple(subject_files))
    return training_data_files


'''
# Some code to construct generators for training, validation and test #
training_data_files = fetch_training_data_files()
write_data_to_file(training_data_files, config["data_file"], image_shape=config["image_shape"])
'''



#plot_model(model, 'slurm_test.png', show_shapes=True)


def main(overwrite=False):
    # convert input images into an hdf5 file
    # if we want to overwrite files or it doen't exist create datafile
    if overwrite or not os.path.exists(config["data_file"]):
        print('fetching data files... \n')
        training_files = fetch_training_data_files()
        print('writing data to file... \n')
        write_data_to_file(training_files, config["data_file"], image_shape=config["image_shape"])
        
    data_file_opened = open_data_file(config["data_file"])
    
    if not overwrite and os.path.exists(config["model_file"]):
        model = load_model(config["model_file"], {'tf' :tf})
        
    else:
        # instantiate new model
        with tf.device('/cpu:0'):
            model = unet_model_3d(input_shape = config["input_shape"], 
                      strided_conv_size = config["strided_conv_size"],
                      n_labels = config["n_labels"],
                      dilation_block = config["dilation_block"],
                      n_dil_block = config["n_dil_block"],
                      initial_learning_rate = config["initial_learning_rate"],
                      deconvolution = config["deconvolution"],
                      #include_label_wise_dice_coefficients = True,
                      residual = config["residual"],
                      dense = config["dense"],
                      n_base_filters = config["n_base_filters"])

    parallel_model = multi_gpu_model(model, gpus=4)
    parallel_model.compile(optimizer=Adam(lr=config['initial_learning_rate']), loss=dice_coefficient_loss, metrics=[dice_coefficient])
    '''  
    # print summary of model to double check, as well as save image of model
    #'''
    parallel_model.summary()
    #plot_model(model, 'model_test.png', show_shapes=True)     
    #'''
    # get training and testing generators
    train_generator, validation_generator, n_train_steps, n_validation_steps = get_training_and_validation_generators(
        data_file_opened,
        batch_size=config["batch_size"],
        data_split=config["validation_split"],
        overwrite=overwrite,
        validation_keys_file=config["validation_file"],
        training_keys_file=config["training_file"],
        n_labels=config["n_labels"],
        labels=config["labels"],
        patch_shape=config["patch_shape"],
        validation_batch_size=config["validation_batch_size"],
        validation_patch_overlap=config["validation_patch_overlap"],
        training_patch_start_offset=config["training_patch_start_offset"],
        #permute=config["permute"],
        #augment=config["augment"],
        skip_blank=config["skip_blank"] #,
        #augment_flip=config["flip"]
        #augment_distortion_factor=config["distort"]
        )

    # run training
    # train the model:
    print('fitting model...')
    parallel_model.fit_generator(generator=train_generator,
                        steps_per_epoch=n_train_steps,
                        epochs=config['n_epochs'],
                        validation_data=validation_generator,
                        validation_steps=n_validation_steps,
                        # Callbacks go to training.log
                        callbacks=get_callbacks(config["model_file"],
                                                logging_file = config['logging_file'],
                                                initial_learning_rate=config["initial_learning_rate"],
                                                learning_rate_drop=config["learning_rate_drop"],
                                                learning_rate_epochs=config["learning_rate_epochs"],
                                                learning_rate_patience=config["patience"],
                                                early_stopping_patience=config["early_stop"]))
    data_file_opened.close()


if __name__ == "__main__":
    main(overwrite=config["overwrite"])
    '''
    prediction_dir = os.path.join(home_path, 'prediction')
    print('running validation cases...')
    run_validation_cases(validation_keys_file=config["validation_file"],
                         GPU = True,
                         model_file=config['model_file'],
                         training_modalities=config["training_modalities"],
                         labels=config["labels"],
                         hdf5_file=config['data_file'],
                         output_label_map=True,
                         custom = True,
                         output_dir=prediction_dir)
    
    custom_objects = {'dice_coefficient_loss': dice_coefficient_loss, 
                      'dice_coefficient': dice_coefficient,
                      'tf' : tf}
    model = load_model(config["model_file"], custom_objects)
    model.summary()
    
    import nibabel as nib
    import numpy as np
    tse = nib.load('/scratch/cai/DEEPSEACAT/data/data_config_flipped/020_left_flipped/flipped_tse_left.nii.gz')
    tse = tse.get_fdata()
    mprage = nib.load('/scratch/cai/DEEPSEACAT/data/data_config_flipped/020_left_flipped/flipped_mprage_left.nii.gz')
    mprage = mprage.get_fdata()
    
    data = np.stack((tse, mprage))
    model.predict(data)    
    '''