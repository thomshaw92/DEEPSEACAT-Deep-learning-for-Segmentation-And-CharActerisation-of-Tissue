#!/usr/bin/env python3
#DEEPSEACAT preprocessing pipeline in nipype
#27/9/19

from os.path import join as opj
import os
from nipype.interfaces.base import (TraitedSpec,
	                            CommandLineInputSpec, 
	                            CommandLine, 
	                            File, 
	                            traits)
from nipype.interfaces.c3 import C3d
from nipype.interfaces.fsl.preprocess import FLIRT
from nipype.interfaces.utility import IdentityInterface#, Function
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.pipeline.engine import Workflow, Node, MapNode
from nipype.interfaces.ants import Registration, RegistrationSynQuick
from nipype.interfaces.ants import ApplyTransforms

os.environ["FSLOUTPUTTYPE"] = "NIFTI_GZ"

#setup for Workstations
#experiment_dir = '/winmounts/uqdlund/uq-research/DEEPSEACAT-Q1219/data/'
experiment_dir = '/data/fastertemp/uqmtottr/'
#where all the atlases live
#github_dir = '/data/home/uqmtottr/DEEPSEACAT-Deep-learning-for-Segmentation-And-CharActerisation-of-Tissue/lib/'
#github_dir = '/winmounts/uqdlund/uq-research/DEEPSEACAT-Q1219/data/DEEPSEACAT_atlas/'
github_dir = '/data/fastertemp/uqmtottr/DEEPSEACAT_atlas/'
##############
#the outdir
output_dir = 'output_dir'
#working_dir name
working_dir = 'Nipype_working_dir_20191011_UMC'
#other things to be set up
side_list = ['right', 'left']

shorter_list = sorted(os.listdir(experiment_dir+'ashs_atlas_umcutrecht/train/')) 
#####################

wf = Workflow(name='Workflow_preprocess_DL_hippo') 
wf.base_dir = os.path.join(experiment_dir+working_dir)

# create infosource to iterate over iterables
infosource = Node(IdentityInterface(fields=['shorter_id',
                                            'side_id']),
                  name="infosource")
infosource.iterables = [('shorter_id', shorter_list),
                        ('side_id', side_list)]


templates = {#tse
             'umc_tse_native' : 'ashs_atlas_umcutrecht/train/{shorter_id}/tse_native_chunk_{side_id}.nii.gz',
             'umc_tse_whole' : 'ashs_atlas_umcutrecht/train/{shorter_id}/tse.nii.gz',
             #seg
             'umc_seg_native' : 'ashs_atlas_umcutrecht/train/{shorter_id}/tse_native_chunk_{side_id}_seg.nii.gz',
             #mprage
             'umc_mprage_chunk' : 'ashs_atlas_umcutrecht/train/{shorter_id}/mprage_to_chunktemp_{side_id}.nii.gz',
             }
# change and add more strings to include all necessary templates
bespoke_files = {'mprage_inthist_template' : '{side_id}_mprage_template_resampled-0.35mmIso_rescaled_0meanUv_pad-176x144x128.nii.gz',
                 'tse_inthist_template' : '{side_id}_tse_template_resampled-0.35mmIso_rescaled_0meanUv_pad-176x144x128.nii.gz'
                 }

selectfiles = Node(SelectFiles(templates, base_directory=experiment_dir), name='selectfiles')

selecttemplates = Node(SelectFiles(bespoke_files, base_directory=github_dir), name='selecttemplates')

wf.connect([(infosource, selectfiles, [('shorter_id', 'shorter_id'),
                                       ('side_id', 'side_id')])]) 

wf.connect([(infosource, selecttemplates, [('side_id','side_id')])])

## Overview of new plan ##
# Step 1 Trim and pad the umc_TSE_native chunks
# Step 2 Reslice umc_TSE_native chunks
# Step 3 Trim and pad the umc segmentation (Maybe we should do the multilabel split here as well, see step 12)
# Step 4 Use ANTS to register umc_MPRAGE to umc_TSE_native chunks
# Step 5 Reslice/pad umc_MPRAGE
# Step 6 Normalize to respective TSE and MPRAGE templates

## DONE WITH UMC ##

# Step 7 Use ANTS to register mag_MPRAGE to mag_TSE_native chunks
# Step 8 Use ANTS to register mag_TSE to umc_TSE
# Step 9a Apply transforms from 8 to mag_TSE_native chunks and mag_TSE_seg_native chunks
# Step 9b Apply transforms from 7 and 8 to mag_MPRAGE
# Step 10 Trim and pad the mag_TSE_native chunks
# Step 11 Reslice mag_TSE_native chunks
# Step 12 Reslice segmentation one label at a time using c3d -split and -merge commands 
# Step 13 Reslice/pad umc_MPRAGE (Can be moved to step 10 if just padding suffices
# Step 14 Normalize to respective TSE and MPRAGE templates

#### DONE WITH MAG ####

# Step 15 Datasink

############
## Step 1 ##
############

#### IMPORTANT - FIX ME ##### New discovery! Some of the tse images in the umc data set have sizes larger than 176x144x128, and therefor we might need to cut these images instead of padding them

#Pad and trim the tse_native_chunk to the correct size 
UMC_TSE_trim_pad_n = MapNode(C3d(interp = "Sinc", pix_type = 'float', args = '-trim-to-size 176x144x128vox -pad-to 176x144x128 0' , out_files = 'UMC_TSE_trim_pad.nii.gz'), #-resample-mm 0.35x0.35x0.35mm 
                            name='UMC_TSE_trim_pad_n', iterfield=['in_file']) 
wf.connect([(selectfiles, UMC_TSE_trim_pad_n, [('umc_tse_native','in_file')])])


###############
## Step 2    ##
###############
#Reslice the trimmed and padded image by the original TSE to get new same sized chunks across the dataset.
UMC_TSE_reslice_n =  MapNode(C3d(interp = "Sinc", pix_type = 'float', args = '-reslice-identity', out_files = 'UMC_TSE_native_resliced.nii.gz'),
                             name='UMC_TSE_reslice_n', iterfield =['in_file'])

# Two inputs needed.
# The image we want to use as a reslicer
wf.connect([(UMC_TSE_trim_pad_n, UMC_TSE_reslice_n, [('out_files','in_file') ])])
# The image to be resliced
wf.connect([(selectfiles, UMC_TSE_reslice_n, [('umc_tse_whole','opt_in_file')])])




############    
## Step 3 ##
############
# Trim and pad the segmentation chunk from UMC_seg_native (UMC_TSE_SEG_native_chunk)

UMC_SEG_trim_pad_n = MapNode(C3d(interp = "NearestNeighbor", pix_type = 'float', args = '-trim-to-size 176x144x128vox -pad-to 176x144x128 0', out_files = 'UMC_TSE_SEG_trim_pad.nii.gz'),#-resample-mm 0.35x0.35x0.35mm 
                                     name='UMC_SEG_trim_pad_n', iterfield=['in_file'])
wf.connect([(selectfiles, UMC_SEG_trim_pad_n, [('umc_seg_native','in_file')])])


############    
## Step 4 ##
############
# ANTS the umc_MPRAGE to the umc_TSE_native

UMC_register_MPRAGE_to_UMC_TSE_native_n = MapNode(Registration(num_threads=30,
                                                               dimension = 3,
                                                               float = False, #False
                                                               interpolation = 'BSpline',
                                                               use_histogram_matching = False, #False
                                                               transforms = ['Rigid', 'Affine', 'SyN'],
                                                               transform_parameters = [[0.2],[0.15],[0.1,3,0.0]],
                                                               metric = ['MI','MI','CC'],
                                                               metric_weight = [1]*3,
                                                               radius_or_number_of_bins = [32, 32, 4],
                                                               sampling_strategy = ['Regular', 'Regular', None],
                                                               sampling_percentage = [0.25,0.25, None],
                                                               number_of_iterations = [[1000,500,250,100], [1000,500,250,100], [100,70,50,20]],
                                                               convergence_threshold = [1e-6]*3,
                                                               convergence_window_size = [10]*3,  
                                                               shrink_factors = [[8,4,2,1],[8,4,2,1],[8,4,2,1]],
                                                               smoothing_sigmas = [[3,2,1,0],[3,2,1,0],[3,2,1,0]],
                                                               sigma_units = ['vox']*3,
                                                               output_warped_image = 'UMC_register_MPRAGE_to_UMC_TSE_native.nii.gz'
                                                              ), 
                                         name = 'UMC_register_MPRAGE_to_UMC_TSE_native_n', iterfield=['fixed_image', 'moving_image'])

wf.connect([(selectfiles, UMC_register_MPRAGE_to_UMC_TSE_native_n, [('umc_tse_native','fixed_image'),
                                                                    ('umc_mprage_chunk','moving_image')
                                                                    ])])

############
## Step 5 ##
############
# Reslice/pad UMC_MPRAGE after registration
UMC_MPRAGE_reslice_n =  MapNode(C3d(interp = "Sinc", pix_type = 'float', args = '-reslice-identity', out_files = 'UMC_MPRAGE_resliced.nii.gz'),
                             name='UMC_MPRAGE_reslice_n', iterfield =['in_file', 'opt_in_file'])

# Reslicing image
wf.connect([(UMC_TSE_reslice_n, UMC_MPRAGE_reslice_n, [('out_files','in_file')])])
# Image to be resliced
wf.connect([(UMC_register_MPRAGE_to_UMC_TSE_native_n, UMC_MPRAGE_reslice_n, [('warped_image','opt_in_file')])])



'''
Alternative approach:
UMC_MPRAGE_trim_pad_n = MapNode(C3d(interp = "Sinc", pix_type = 'float', args = '-trim-to-size 176x144x128vox -pad-to 176x144x128 0' , out_files = 'UMC_MPRAGE_chunk_trim_pad.nii.gz'),
                            name='UMC_MPRAGE_trim_pad_n', iterfield=['in_file']) 
wf.connect([(selectfiles, UMC_MPRAGE_to_UMC_TSE_native_n, [('warped_image','in_file')])])
'''


############
## Step 6 ##
############
# Normalize UMC and MPRAGE to respective templates

#TSE normalization UMC
UMC_normalise_TSE_n = MapNode(C3d(interp="Sinc", pix_type='float', args='-histmatch 5', out_file = 'UMC_normalize_TSE_native.nii.gz'),
                          name='UMC_normalise_TSE_n', iterfield=['in_file'])
wf.connect([(UMC_TSE_reslice_n, UMC_normalise_TSE_n, [('out_files', 'in_file')])])
wf.connect([(selecttemplates, UMC_normalise_TSE_n, [('tse_inthist_template', 'opt_in_file')])])

#MPRAGE normalization UMC
UMC_normalise_MPRAGE_n = MapNode(C3d(interp="Sinc", pix_type='float', args='-histmatch 5', out_file = 'UMC_normalize_MPRAGE.nii.gz'),
                             name='UMC_normalise_MPRAGE_n', iterfield=['in_file'])
wf.connect([(UMC_MPRAGE_reslice_n, UMC_normalise_MPRAGE_n, [('out_files', 'in_file')])])
wf.connect([(selecttemplates, UMC_normalise_MPRAGE_n, [('mprage_inthist_template', 'opt_in_file')])])



######################################################## END OF UMC PROCESSING ##################################################




################
## DATA SINK  ##
################
datasink = Node(DataSink(base_directory=experiment_dir+working_dir,
                         container=output_dir),
                name="datasink")
#wf.connect([(umc_tse_whole_resample_n, datasink, [('out_files','UMC_TSE_whole_resample')])]) #Step 1
wf.connect([(UMC_TSE_trim_pad_n, datasink, [('out_files','UMC_TSE_native_trim_pad')])]) #Step 1
wf.connect([(UMC_TSE_reslice_n, datasink, [('out_files','UMC_TSE_native_reslice')])]) #Step 2
wf.connect([(UMC_SEG_trim_pad_n, datasink, [('out_files', 'UMC_TSE_SEG_trim_pad')])]) #Step 3
wf.connect([(UMC_register_MPRAGE_to_UMC_TSE_native_n, datasink, [('forward_transforms','UMC_register_MPRAGE_to_UMC_TSE_native_n_forward_transforms')])]) #Step 4
wf.connect([(UMC_register_MPRAGE_to_UMC_TSE_native_n, datasink, [('warped_image','UMC_register_MPRAGE_to_UMC_TSE_native_n')])]) #Step 4
wf.connect([(UMC_MPRAGE_reslice_n, datasink, [('out_files', 'UMC_MPRAGE_reslice_n')])]) #Step 5
wf.connect([(UMC_normalise_TSE_n, datasink, [('out_files', 'UMC_TSE_normalised')])]) #Step 6
wf.connect([(UMC_normalise_MPRAGE_n, datasink, [('out_files','UMC_MPRAGE_normalised')])]) #Step 6


########################################################### END OF UMC DATASINK #################################################

###################
## Run the thing ##
###################
wf.write_graph(graph2use='flat', format='png', simple_form=False)

#wf.run(plugin='SLURMGraph', plugin_args = {'dont_resubmit_completed_jobs': True} )
#wf.run()
wf.run(plugin='MultiProc', plugin_args = {'n_procs' : 30})

'''


# # run as MultiProc
wf.write_graph(graph2use='flat', format='png', simple_form=False)
#wf.run(plugin='SLURMGraph', plugin_args = {'dont_resubmit_completed_jobs': True} )


'''
#This is for running at CAI
# qsub_args='-N 1,-c 4,--partition=all, --mem=16000'))
#running at Awoonga:
