#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 09:53:02 2019

@author: uqdlund
"""

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
###############
# work on scratch space only - We will work on Awoonga because Wiener is GPUs only.
#experiment_dir = '/RDS/Q1219/data/' 
#output_dir = '/RDS/Q1219/data/preprocessed'
#working_dir = '/30days/${USER}/DEEPSEACAT_WORKINGDIR'
#github_dir = '~/DEEPSEACAT-Deep-learning-for-Segmentation-And-CharActerisation-of-Tissue/ '
################
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
working_dir = 'Nipype_working_dir_20191011_MAG'
#other things to be set up
side_list = ['right', 'left']

subject_list = sorted(os.listdir(experiment_dir+'ashs_atlas_magdeburg/train/'))
subject_list = subject_list[0:2]

#####################

wf = Workflow(name='Workflow_preprocess_DL_hippo') 
wf.base_dir = os.path.join(experiment_dir+working_dir)

# create infosource to iterate over iterables
infosource = Node(IdentityInterface(fields=['subject_id',
                                            'side_id']),
                  name="infosource")
infosource.iterables = [('subject_id', subject_list),
                        ('side_id', side_list)]


templates = {#tse
             'mag_tse_native' : 'ashs_atlas_magdeburg/train/{subject_id}/tse_native_chunk_{side_id}.nii.gz',
             'mag_tse_whole' : 'ashs_atlas_magdeburg/train/{subject_id}/tse.nii.gz',
             #seg
             'mag_seg_native' : 'ashs_atlas_magdeburg/train/{subject_id}/tse_native_chunk_{side_id}_seg.nii.gz',
             #mprage
             'mag_mprage_chunk' : 'ashs_atlas_magdeburg/train/{subject_id}/mprage_to_chunktemp_{side_id}.nii.gz',
             }
# change and add more strings to include all necessary templates
bespoke_files = {'umc_tse_whole_template' : 'umc_tse_template_resampled_0.35mm.nii.gz',
                 'mprage_inthist_template' : '{side_id}_mprage_template_resampled-0.35mmIso_rescaled_0meanUv_pad-176x144x128.nii.gz',
                 'tse_inthist_template' : '{side_id}_tse_template_resampled-0.35mmIso_rescaled_0meanUv_pad-176x144x128.nii.gz'
                 }

selectfiles = Node(SelectFiles(templates, base_directory=experiment_dir), name='selectfiles')

selecttemplates = Node(SelectFiles(bespoke_files, base_directory=github_dir), name='selecttemplates')

wf.connect([(infosource, selectfiles, [('subject_id', 'subject_id'),
                                       ('side_id', 'side_id')])]) 

wf.connect([(infosource, selecttemplates, [('side_id','side_id')])])



############
## Step 7 ##
############
# ANTS MPRAGE to UMC_TSE_native_chunk
MAG_register_MPRAGE_to_MAG_native_n = MapNode(Registration(num_threads=30,
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
                                                           output_warped_image = 'MAG_register_MPRAGE_to_MAG_native.nii.gz'
                                                          ), 
                                         name = 'MAG_register_MPRAGE_to_MAG_native_n', iterfield=['fixed_image', 'moving_image'])

wf.connect([(selectfiles, MAG_register_MPRAGE_to_MAG_native_n, [('mag_tse_native','fixed_image'),
                                                                    ('mag_mprage_chunk','moving_image')
                                                                    ])])


############
## Step 8 ##
############
# Register MAG_TSE_whole ---> UMC_TSE_whole 
    # NOTE! THIS IS THE ONLY PLACE WHERE WE ARE OUTPUTTING A WHOLE IMAGE!
# So we register the TSEs of the magdeberg dataset to the template of the UMC dataset rigidly.
# The template is in the github repo cause we had to make it first (see bash script).
    

MAG_register_TSE_whole_to_UMC_TSE_whole_n = MapNode(RegistrationSynQuick(transform_type = 'r', use_histogram_matching=True), 
                         name='MAG_register_TSE_whole_to_UMC_TSE_whole_n', iterfield=['moving_image'])
wf.connect([(selecttemplates, MAG_register_TSE_whole_to_UMC_TSE_whole_n, [('umc_tse_whole_template', 'fixed_image')])])
wf.connect([(selectfiles, MAG_register_TSE_whole_to_UMC_TSE_whole_n, [('mag_tse_whole', 'moving_image')])])


#############
## Step 9a ##
#############
#the mag data needs to have the same treatment as the umc data now.
#First, take the transformation that we just computed and apply it to the mag_tse_native_chunk
MAG_TSE_native_move_n = MapNode(ApplyTransforms(dimension = 3, interpolation = 'BSpline', output_image = 'MAG_TSE_native_move.nii.gz'),
                            name='MAG_TSE_native_move_n', iterfield=['input_image', 'transforms', 'reference_image']) #Hmm, we should need to iterate over transforms as well??

wf.connect([(selectfiles, MAG_TSE_native_move_n, [('mag_tse_native', 'input_image')])])
wf.connect([(MAG_register_TSE_whole_to_UMC_TSE_whole_n, MAG_TSE_native_move_n, [('out_matrix', 'transforms'),
                                                                                ('warped_image','reference_image')])]) 

# then apply the transformation to the mag_tse_native_seg
MAG_SEG_native_move_n = MapNode(ApplyTransforms(dimension = 3, interpolation = 'MultiLabel', output_image = 'MAG_SEG_native_move.nii.gz'), #Nearest Neighbor multilabel transform
                            name='MAG_SEG_native_move_n', iterfield=['input_image', 'transforms', 'reference_image'])

wf.connect([(selectfiles, MAG_SEG_native_move_n, [('mag_seg_native', 'input_image')])])
wf.connect([(MAG_register_TSE_whole_to_UMC_TSE_whole_n, MAG_SEG_native_move_n, [('out_matrix', 'transforms'),
                                                                                ('warped_image','reference_image')])])

#############
## Step 9b ##
#############
#the mag data needs to have the same treatment as the umc data now.
#First, take the transformation that we just computed and apply it to the mag_tse_native_chunk

MAG_MPRAGE_native_move_n = MapNode(ApplyTransforms(dimension = 3, interpolation = 'BSpline', output_image = 'MAG_MPRAGE_native_move.nii.gz'),
                                   name='MAG_MPRAGE_native_move_n', iterfield=['input_image', 'transforms', 'reference_image']) #Hmm, we should need to iterate over transforms as well??

wf.connect([(MAG_register_MPRAGE_to_MAG_native_n, MAG_MPRAGE_native_move_n, [('warped_image', 'input_image')])])
wf.connect([(MAG_register_TSE_whole_to_UMC_TSE_whole_n, MAG_MPRAGE_native_move_n, [('out_matrix', 'transforms'),
                                                                                   ('warped_image','reference_image')])])


# OLD way with 3 applied transforms, NOT WORKING
'''
wf.connect([(selectfiles, MAG_MPRAGE_native_move_n, [('mag_tse_native', 'input_image')])])
wf.connect([(MAG_register_MPRAGE_to_MAG_native_n, MAG_MPRAGE_native_move_n, [('forward_transforms', 'transforms'),
                                                                             ('warped_image','reference_image')
                                                                             ])]) 
wf.connect([(MAG_register_TSE_whole_to_UMC_TSE_whole_n, MAG_MPRAGE_native_move_n, [('out_matrix', 'opt_transforms')])])
'''
# We don't know if the above works currently, the idea is that we save an interpolation step by applying all transforms simultaneously
# instead of sequentially
# Alternatively just apply transformations sequentially


'''
This has been added to resampling.py in ants.interfaces
    opt_transforms = File(
        #InputMultiPath(File(exists=True)),
        argstr='-t %s',
        mandatory=False,
        desc='transform files: will be applied in reverse order. For '
        'example, the last specified transform will be applied first.')
'''

'''
   ${ANTSPATH}antsApplyTransforms \
  -d 3 \
  -i movingImage.nii.gz \
  
  -r fixedImage.nii.gz \
  -t tse_to_tse.mat
  -t movingToFixed_1Warp.nii.gz \
  -t movingToFixed_0GenericAffine.mat \
  
  -o movingToFixedDeformed.nii.gz 
'''





##################
##  Step 10     ##
##################
#Repeat Steps 2-4 but with registered Magdeburg
# Pad and trim the mag tse native chunk
MAG_TSE_trim_pad_n = MapNode(C3d(interp = "Sinc", pix_type = 'float', args = '-trim-to-size 176x144x128vox -pad-to 176x144x128 0' , out_files = 'MAG_TSE_trim_padded.nii.gz'),
                            name='MAG_TSE_trim_pad_n', iterfield=['in_file']) 
wf.connect([(MAG_TSE_native_move_n, MAG_TSE_trim_pad_n, [('output_image','in_file')])])


##################
##  Step 11     ##
##################
# Reslice the whole mag tse image to a new chunk based on the mag_tse_resample_pad_bin
MAG_TSE_reslice_n =  MapNode(C3d(interp = "Sinc", pix_type = 'float', args = '-reslice-identity', out_files = 'MAG_TSE_resliced.nii.gz'),
                             name='MAG_TSE_reslice_n', iterfield =['in_file','opt_in_file'])

wf.connect([(MAG_TSE_trim_pad_n, MAG_TSE_reslice_n, [('out_files','in_file') ])])
wf.connect([(MAG_TSE_native_move_n, MAG_TSE_reslice_n, [('output_image','opt_in_file') ])])


##################
##  Step 12     ##
##################
# Reslice labels by splitting labels into binary images and reslicing them individually

MAG_SEG_reslice_labels = MapNode(C3d(interp = "NearestNeighbor",
                                     args =  ' -split ' +
                                             '-foreach ' +
                                             '-insert ref 1 ' +
                                             '-reslice-identity ' +
                                             '-endfor ' +
                                             '-merge', 
                                     out_files = 'MAG_SEG_resliced_labels.nii.gz'),
                            name='MAG_SEG_reslice_labels', iterfield=['in_file', 'ref_in_file']) 
# Image used to reslice
wf.connect([(MAG_TSE_reslice_n, MAG_SEG_reslice_labels, [('out_files','in_file')])])
# Image to be resliced, NOTE ref_in_file
wf.connect([(MAG_SEG_native_move_n, MAG_SEG_reslice_labels, [('output_image','ref_in_file')])])


'''
# Second we resample and pad the mag_seg_native 
wf.connect([(MAG_register_n, MAG_TSE_reslice_n, [('warped_image','opt_in_file')])])
##### THIS IS WHERE WE FUCK UP! IS IT INTERPOLATION???? ##### Used to be Sinc
mag_seg_trim_pad_n = MapNode(C3d(interp = "NearestNeighbor", pix_type = 'float', args = '-trim-to-size 176x144x128vox -pad-to 176x144x128 0', out_files = 'mag_chunk_seg_trim_padded.nii.gz'),#-resample-mm 0.35x0.35x0.35mm
                                     name='mag_seg_trim_pad_n', iterfield=['in_file'])
wf.connect([(MAG_SEG_native_move_n, MAG_SEG_trim_pad_n, [('output_image','in_file')])])

'''

##################
##  Step 13     ##
##################
# Reslice MAG_MPRAGE

MAG_MPRAGE_reslice_n =  MapNode(C3d(interp = "Sinc", pix_type = 'float', args = '-reslice-identity', out_files = 'MAG_MPRAGE_resliced.nii.gz'),
                             name='MAG_MPRAGE_reslice_n', iterfield =['in_file', 'opt_in_file'])

# Reslicing image
wf.connect([(MAG_TSE_reslice_n, MAG_MPRAGE_reslice_n, [('out_files','in_file')])])
# Image to be resliced
wf.connect([(MAG_MPRAGE_native_move_n, MAG_MPRAGE_reslice_n, [('output_image','opt_in_file')])])

##############
##  Step 14  ##
##############
#Normalization of the images to the templates for the overall tse/mprage intensity histogram
#mprage normalization magdeburg
MAG_normalise_MPRAGE_n = MapNode(C3d(interp="Sinc", pix_type='float', args='-histmatch 5' , out_files = 'MAG_normalise_MPRAGE_n.nii.gz'),
                             name='MAG_normalise_MPRAGE_n', iterfield=['in_file'])
wf.connect([(MAG_MPRAGE_reslice_n, MAG_normalise_MPRAGE_n, [('out_files', 'in_file')])])
wf.connect([(selecttemplates, MAG_normalise_MPRAGE_n, [('mprage_inthist_template', 'opt_in_file')])])


#tse normalization magdeburg
MAG_normalise_TSE_n = MapNode(C3d(interp="Sinc", pix_type='float', args='-histmatch 5', out_file = 'MAG_normalise_TSE.nii.gz'),
                          name='MAG_normalise_TSE_n', iterfield=['in_file'])
wf.connect([(MAG_TSE_reslice_n, MAG_normalise_TSE_n, [('out_files', 'in_file')])])
wf.connect([(selecttemplates, MAG_normalise_TSE_n, [('tse_inthist_template', 'opt_in_file')])])


################
## DATA SINK  ##
################
datasink = Node(DataSink(base_directory=experiment_dir+working_dir,
                         container=output_dir),
                name="datasink")

wf.connect([(MAG_register_MPRAGE_to_MAG_native_n, datasink, [('warped_image','MAG_register_MPRAGE_to_MAG_native_n')])]) #Step 7
wf.connect([(MAG_register_TSE_whole_to_UMC_TSE_whole_n, datasink, [('warped_image','MAG_register_TSE_whole_to_UMC_TSE_whole')])]) #Step 8
wf.connect([(MAG_TSE_native_move_n, datasink, [('output_image','MAG_TSE_native_move_n')])]) #Step 9a
wf.connect([(MAG_SEG_native_move_n, datasink, [('output_image','MAG_SEG_native_move_n')])]) #Step 9a
wf.connect([(MAG_MPRAGE_native_move_n, datasink, [('output_image','MAG_MPRAGE_native_move_n')])]) #Step 9b
wf.connect([(MAG_TSE_trim_pad_n, datasink, [('out_files','MAG_TSE_trim_pad')])]) #Step 10
wf.connect([(MAG_TSE_reslice_n, datasink, [('out_files','MAG_TSE_reslice_n')])]) #Step 11
wf.connect([(MAG_SEG_reslice_labels, datasink, [('out_files','MAG_SEG_reslice_labels')])]) #Step 12
wf.connect([(MAG_MPRAGE_reslice_n, datasink, [('out_files','MAG_MPRAGE_reslice_n')])]) #Step 13
wf.connect([(MAG_normalise_TSE_n, datasink, [('out_files', 'MAG_TSE_normalised')])]) #Step 14
wf.connect([(MAG_normalise_MPRAGE_n, datasink, [('out_files','MAG_MPRAGE_normalised')])]) #Step 14



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