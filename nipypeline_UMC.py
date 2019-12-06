#!/usr/bin/env python3
#DEEPSEACAT preprocessing pipeline in nipype
#27/9/19

import os
from Model.config import src_path
from Preprocessing.c3 import C3d
from nipype.interfaces.utility import IdentityInterface#, Function
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.pipeline.engine import Workflow, Node, MapNode
from nipype.interfaces.ants import RegistrationSynQuick

os.environ["FSLOUTPUTTYPE"] = "NIFTI_GZ"

#where all the atlases live
atlas_dir = os.path.join(os.getcwd(),'DEEPSEACAT_atlas')
##############
#the outdir
output_dir = 'UMC_output'
#working_dir name
working_dir = 'Nipype'
#other things to be set up
side_list = ['right', 'left']
#the subject list, here called shorter list because the UMC dataset contains less subjects than the magdeburge dataset
shorter_list = sorted(os.listdir(src_path+'ashs_atlas_umcutrecht/train/')) 
#shorter_list = shorter_list[1:2]
#####################

wf = Workflow(name='UMC_workflow') 
wf.base_dir = os.path.join(src_path+working_dir)

# create infosource to iterate over iterables
infosource = Node(IdentityInterface(fields=['subject_id',
                                            'side_id']),
                  name="infosource")
infosource.iterables = [('shorter_id', shorter_list),
                        ('side_id', side_list)]

# Different images used in this pipeline
templates = {#tse
             'umc_tse_native' : 'ashs_atlas_umcutrecht/train/{shorter_id}/tse_native_chunk_{side_id}.nii.gz',
             'umc_tse_whole' : 'ashs_atlas_umcutrecht/train/{shorter_id}/tse.nii.gz',
             #seg
             'umc_seg_native' : 'ashs_atlas_umcutrecht/train/{shorter_id}/tse_native_chunk_{side_id}_seg.nii.gz',
             #mprage
             'umc_mprage_chunk' : 'ashs_atlas_umcutrecht/train/{shorter_id}/mprage_to_chunktemp_{side_id}.nii.gz',
             }

# Different templates used in this pipeline
bespoke_files = {'mprage_inthist_template' : '{side_id}_mprage_template_resampled-0.35mmIso_rescaled_0meanUv_pad-176x144x128.nii.gz',
                 'tse_inthist_template' : '{side_id}_tse_template_resampled-0.35mmIso_rescaled_0meanUv_pad-176x144x128.nii.gz'
                 }

selectfiles = Node(SelectFiles(templates, base_directory=src_path), name='selectfiles')

selecttemplates = Node(SelectFiles(bespoke_files, base_directory=atlas_dir), name='selecttemplates')

wf.connect([(infosource, selectfiles, [('shorter_id', 'shorter_id'),
                                       ('side_id', 'side_id')])]) 

wf.connect([(infosource, selecttemplates, [('side_id','side_id')])])

## Overview of the UMC preprocessing steps ##
# Step 1 Trim and pad the umc_TSE_native chunks
# Step 2 Reslice umc_TSE_native chunks
# Step 3 Reslice the umc segmentation
# Step 4 Register umc_MPRAGE to umc_TSE_native chunks
# Step 5 Reslice umc_MPRAGE
# Step 6 Normalize TSE and MPRAGE images to respective TSE and MPRAGE templates
# Datasink

############
## Step 1 ##
############
#Pad and trim the tse_native_chunk to the correct size 
UMC_trim_pad_TSE_n = MapNode(C3d(interp = "Sinc", pix_type = 'float', args = '-trim-to-size 176x144x128vox -pad-to 176x144x128 0' , out_files = 'UMC_TSE_trim_pad.nii.gz'), 
                            name='UMC_trim_pad_TSE_n', iterfield=['in_file']) 

wf.connect([(selectfiles, UMC_trim_pad_TSE_n, [('umc_tse_native','in_file')])])


#############
## Step 2 ##
#############
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