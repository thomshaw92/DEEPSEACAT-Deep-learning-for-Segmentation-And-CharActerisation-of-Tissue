#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 11:34:00 2020

@author: uqtshaw
"""

#!/usr/bin/env python3
#DEEPSEACAT preprocessing pipeline in nipype
#27/9/19

import os
#from Model.config import src_path
from c3 import C3d
from nipype.interfaces.utility import IdentityInterface #, Function
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.pipeline.engine import Workflow, Node, MapNode
from nipype.interfaces.ants import RegistrationSynQuick, N4BiasFieldCorrection, DenoiseImage, ApplyTransforms 

os.environ["FSLOUTPUTTYPE"] = "NIFTI_GZ"

#where all the atlases live
src_path = '/data/lfs2/uqtshaw/DEEPSEACAT/'
atlas_dir = os.path.join(os.getcwd(),'ADNI_atlas')
##############
#the outdir
output_dir = 'ADNI_output'
#working_dir name
working_dir = 'Nipype'
#other things to be set up
#side_list = ['right', 'left']
#the subject list, here called shorter list because the UMC dataset contains fewer subjects than the magdeburg dataset
shorter_list = sorted(os.listdir(src_path+'bids_pruned_20200928/sub*')) 
#####################

wf = Workflow(name='ADNI_workflow') 
wf.base_dir = os.path.join(src_path+working_dir)

# create infosource to iterate over iterables
infosource = Node(IdentityInterface(fields=['subject_id',
                                            'side_id']),
                  name="infosource")
infosource.iterables = [('shorter_id', shorter_list),
                        ('side_id', side_list)]

# Different images used in this pipeline
templates = {#tse
             't1w' : 'bids_pruned_20200928/{shorter_id}/ses-01/anat/*run-1_T1w.nii.gz',
             't2w' : 'bids_pruned_20200928/{shorter_id}/ses-01/anat/*run-1_T2w.nii.gz',
             }

# Different templates used in this pipeline
bespoke_files = {'mprage_adni_template' : 'correct_spacing_t1w_adni_atlas_ashs_space.nii.gz',
                 #'tse_adni_template' : '{side_id}_tse_template_resampled-0.35mmIso_rescaled_0meanUv_pad-176x144x128.nii.gz',
                 'mprage_template_bin_chunk_{side_id}' : 'refspace_{side_id}_0.5mm_112x144x124_bin.nii.gz' ##this may be in the wrong space? 
                 }

selectfiles = Node(SelectFiles(templates, base_directory=src_path), name='selectfiles')

selecttemplates = Node(SelectFiles(bespoke_files, base_directory=atlas_dir), name='selecttemplates')

wf.connect([(infosource, selectfiles, [('shorter_id', 'shorter_id'),
                                       ('side_id', 'side_id')])]) 

wf.connect([(infosource, selecttemplates, [('side_id','side_id')])])

## Overview of the ADNI preprocessing steps ##
#0: N4 the T1 and T2
#1: Register the T1w to the template (non lin)
#2: Reverse the flow field and affine for the template chunk mask
#3: register the T2w to the T1w #### -d 3 -a -dof 6 -m MI -n 100x100x10 \
#4: multiply input T1w by chunk mask in subject space
#5: interpolate TSE 
#6: invert the warp from step 3 to bring masked chunk from step 4 into T2 space (resample as well into TSE)
#7: invert the warp from step 3 to bring T1w mask from step 4 into T2 space 
#8: multiply resampled TSE from step 5 by chunk from step 7
#9: denoise chunks
#10: pad and trim chunks and resize
#11: Normalise TSE and MPRAGE chunks respective to the templates
#12: Datasink 


############
## Step 0 ##
############
#N4 
N4_T1_n = MapNode(N4BiasFieldCorrection(n4.inputs.dimension = 3, n4.inputs.bspline_fitting_distance = 300, n4.inputs.bspline_fitting_distance = 300, n4.inputs.shrink_factor = 3, n4.inputs.n_iterations = [50,50,30,20]),
                  name='N4_T1_n', iterfield=['in_file'])
N4_T2_n = MapNode(N4BiasFieldCorrection(n4.inputs.dimension = 3, n4.inputs.bspline_fitting_distance = 300, n4.inputs.bspline_fitting_distance = 300, n4.inputs.shrink_factor = 3, n4.inputs.n_iterations = [50,50,30,20]),
                  name='N4_T1_n', iterfield=['in_file'])

wf.connect([(selectfiles, N4_t1_n, [('t1w','in_file')])])
wf.connect([(selectfiles, N4_t2_n, [('t2w','in_file')])])

##output_image
#############
## Step 1 ##
#############
#register the T1w to the template.
register_T1w_to_template_n = MapNode(RegistrationSynQuick(transform_type = 's'),
                                name='register_T1w_to_template_n', iterfield=['fixed_image', 'moving_image'])
wf.connect([(N4_t1_n, register_T1w_to_template_n, [('output_image', 'moving_image')])])
wf.connect([(selecttemplates, register_T1w_to_template_n, [('mprage_adni_template', 'fixed_image')])])

#############
## Step 2 ##
#############
#Reverse the flow field and affine for the template chunk mask to the input T1w 

invert_chunk_to_T1w_n = MapNode(ApplyTransforms(interpolation = 'NearestNeighbor', ),
                                name='invert_chunk_to_T1w_n', iterfield=['input_image', 'reference_image'])
wf.connect([(register_T1w_to_template_n, invert_chunk_to_T1w_n, [('output_image', 'moving_image', 'transforms')])])
wf.connect([(selecttemplates, invert_chunk_to_T1w_n, [('mprage_adni_template_chunk_{side_id}', 'moving_image')])])
wf.connect([(register_T1w_to_template_n, invert_chunk_to_T1w_n, [('out_matrix' 'transforms')])])

at = ApplyTransforms()
at.inputs.dimension = 3
at.inputs.input_image = 'moving1.nii'
at.inputs.reference_image = 'fixed1.nii'
at.inputs.output_image = 'deformed_moving1.nii'
at.inputs.interpolation = 'Linear'
at.inputs.default_value = 0
at.inputs.transforms = ['ants_Warp.nii.gz', 'trans.mat']
at.inputs.invert_transform_flags = [False, True]


#############
## Step 3 ##
#############
#register the t2w to the t1w
register_T2w_to_T1w_n = MapNode(RegistrationSynQuick(transform_type = 's', ),
                     	name='register_T2w_to_T1w_n', iterfield=['fixed_image', 'moving_image'])

wf.connect([(N4_t1_n, register_T2w_to_T1w_n, [('output_image', 'fixed_image')])])
wf.connect([(N4_t2_n, register_T2w_to_T1w_n, [('output_image', 'moving_image')])])


#forward_warp_field (a pathlike object or string representing an existing file) – Forward warp field.

#inverse_warp_field (a pathlike object or string representing an existing file) – Inverse warp field.

#inverse_warped_image (a pathlike object or string representing an existing file) – Inverse warped image.

#out_matrix (a pathlike object or string representing an existing file) – Affine matrix.

#warped_image (a pathlike object or string representing an existing file) – Warped image.

#############
## Step 4 ##
#############




################ end 


#Pad and trim the tse_native_chunk to the correct size 
UMC_trim_pad_TSE_n = MapNode(C3d(interp = "Sinc", pix_type = 'float', args = '-trim-to-size 176x144x128vox -pad-to 176x144x128 0' , out_files = 'UMC_TSE_trim_pad.nii.gz'), 
                            name='UMC_trim_pad_TSE_n', iterfield=['in_file']) 

wf.connect([(selectfiles, UMC_trim_pad_TSE_n, [('umc_tse_native','in_file')])])




### denoise 

#Reslice the trimmed and padded image by the original TSE to get new same sized chunks across the dataset.
UMC_reslice_TSE_n =  MapNode(C3d(interp = "Sinc", pix_type = 'float', args = '-reslice-identity', out_files = 'UMC_TSE_native_resliced.nii.gz'),
                             name='UMC_reslice_TSE_n', iterfield =['in_file', 'opt_in_file'])

# Two inputs needed.
# The image we want to use as a reslicer
wf.connect([(UMC_trim_pad_TSE_n, UMC_reslice_TSE_n, [('out_files','in_file') ])])
# The image to be resliced
wf.connect([(selectfiles, UMC_reslice_TSE_n, [('umc_tse_whole','opt_in_file')])])


############    
## Step 3 ##
############
# Reslice the segmentation chunk from UMC_seg_native (UMC_TSE_SEG_native_chunk) using multilabel split
UMC_reslice_labels_SEG_n = MapNode(C3d(interp = "NearestNeighbor",
                                     args =  ' -split ' +
                                             '-foreach ' +
                                             '-insert ref 1 ' +
                                             '-reslice-identity ' +
                                             '-endfor ' +
                                             '-merge', 
                                     out_files = 'UMC_SEG_resliced_labels.nii.gz'),
                            name='UMC_reslice_labels_SEG_n', iterfield=['in_file', 'ref_in_file']) 

# Image used to reslice
wf.connect([(UMC_reslice_TSE_n, UMC_reslice_labels_SEG_n, [('out_files','in_file')])])
# Image to be resliced, NOTE ref_in_file needs to be added in the c3 interface!
wf.connect([(selectfiles, UMC_reslice_labels_SEG_n, [('umc_seg_native','ref_in_file')])])


############    
## Step 4 ##
############
# ANTS is used to register the umc_MPRAGE to the umc_TSE_native
UMC_register_MPRAGE_to_UMC_TSE_native_n = MapNode(RegistrationSynQuick(transform_type = 'a'),
                     	name='UMC_register_MPRAGE_to_UMC_TSE_native_n', iterfield=['fixed_image', 'moving_image'])

wf.connect([(selectfiles, UMC_register_MPRAGE_to_UMC_TSE_native_n, [('umc_tse_native', 'fixed_image'),
                                                                    ('umc_mprage_chunk', 'moving_image')])])


############
## Step 5 ##
############
# Reslice UMC_MPRAGE after registration
UMC_reslice_MPRAGE_n =  MapNode(C3d(interp = "Sinc", pix_type = 'float', args = '-reslice-identity', out_files = 'UMC_MPRAGE_resliced.nii.gz'),
                             name='UMC_reslice_MPRAGE_n', iterfield =['in_file', 'opt_in_file'])

# Reslicing image
wf.connect([(UMC_reslice_TSE_n, UMC_reslice_MPRAGE_n, [('out_files','in_file')])])
# Image to be resliced
wf.connect([(UMC_register_MPRAGE_to_UMC_TSE_native_n, UMC_reslice_MPRAGE_n, [('warped_image','opt_in_file')])])


############
## Step 6 ##
############
# Normalize UMC and MPRAGE to respective templates

#TSE normalisation
UMC_normalize_TSE_n = MapNode(C3d(interp="Sinc", pix_type='float', args='-histmatch 5', out_file = 'UMC_normalize_TSE_native.nii.gz'),
                          name='UMC_normalize_TSE_n', iterfield=['in_file'])

wf.connect([(selecttemplates, UMC_normalize_TSE_n, [('tse_inthist_template', 'in_file')])])
wf.connect([(UMC_reslice_TSE_n, UMC_normalize_TSE_n, [('out_files', 'opt_in_file')])])

#MPRAGE normalisation
UMC_normalize_MPRAGE_n = MapNode(C3d(interp="Sinc", pix_type='float', args='-histmatch 5', out_file = 'UMC_normalize_MPRAGE.nii.gz'),
                             name='UMC_normalize_MPRAGE_n', iterfield=['in_file'])

wf.connect([(selecttemplates, UMC_normalize_MPRAGE_n, [('mprage_inthist_template', 'in_file')])])
wf.connect([(UMC_reslice_MPRAGE_n, UMC_normalize_MPRAGE_n, [('out_files', 'opt_in_file')])])


######################################################## END OF UMC PROCESSING ##################################################


################
## DATA SINK  ##
################
datasink = Node(DataSink(base_directory=src_path+working_dir,
                         container=output_dir),
                name="datasink")

wf.connect([(UMC_reslice_labels_SEG_n, datasink, [('out_files', 'UMC_reslice_labels_SEG')])])
wf.connect([(UMC_normalize_TSE_n, datasink, [('out_files', 'UMC_normalized_TSE')])]) #Step 6
wf.connect([(UMC_normalize_MPRAGE_n, datasink, [('out_files','UMC_normalized_MPRAGE')])]) #Step 6


########################################################### END OF UMC DATASINK #################################################


##########
## Run ##
#########
wf.write_graph(graph2use='flat', format='png', simple_form=False)

wf.run()
#wf.run(plugin='SLURMGraph', plugin_args = {'dont_resubmit_completed_jobs': True} )
#wf.run(plugin='MultiProc', plugin_args = {'n_procs' : 30})
