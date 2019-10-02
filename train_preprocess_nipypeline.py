#!/usr/bin/env python3
#DEEPSEACAT preprocessing pipeline in nipype
#27/9/19

from os.path import join as opj
import os
from nipype.interfaces.base import (TraitedSpec,
	                            CommandLineInputSpec, 
	                            CommandLine, 
	                            File, 
	                            traits
)

from nipype.interfaces.c3 import C3d
from nipype.interfaces.fsl.preprocess import FLIRT
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.pipeline.engine import Workflow, Node, MapNode
from nipype.interfaces.ants import RegistrationSynQuick
from nipype.interfaces.ants import ApplyTransforms


#test git
#from nipype import config
#config.enable_debug_mode()
#config.set('execution', 'stop_on_first_crash', 'true')
#config.set('execution', 'remove_unnecessary_outputs', 'false')
#config.set('execution', 'keep_inputs', 'true')
#config.set('logging', 'workflow_level', 'DEBUG')
#config.set('logging', 'interface_level', 'DEBUG')
#config.set('logging', 'utils_level', 'DEBUG')

os.environ["FSLOUTPUTTYPE"] = "NIFTI_GZ"
###############
# work on scratch space only - We will work on Awoonga because Wiener is GPUs only.
#experiment_dir = '/RDS/Q1219/data/' 
#output_dir = '/RDS/Q1219/data/preprocessed'
#working_dir = '/30days/${USER}/DEEPSEACAT_WORKINGDIR'
#github_dir = '~/DEEPSEACAT-Deep-learning-for-Segmentation-And-CharActerisation-of-Tissue/ '
################
#setup for Workstations
experiment_dir = '/winmounts/uqdlund/uq-research/DEEPSEACAT-Q1219/data/'
#where all the atlases live
data_dir = '/data/fastertemp/uqtshaw/'
#files from github that need to be included in the distribution
github_dir = '/data/home/uqdlund/DEEPSEACAT-Deep-learning-for-Segmentation-And-CharActerisation-of-Tissue/lib/'
###############
#the outdir
output_dir = 'output_dir'
#working_dir name
working_dir = 'Nipype_working_dir'
#other things to be set up
dataset_list = ['umcutrecht']
side_list = ['right']
subject_list = ['train000', 'train001']
#####################

wf = Workflow(name='Workflow_preprocess_DL_hippo') 
wf.base_dir = os.path.join(experiment_dir+working_dir)

# create infosource to iterate over iterables
infosource = Node(IdentityInterface(fields=['subject_id',
                                            'side_id',
                                            'dataset_id']),
                  name="infosource")
infosource.iterables = [('subject_id', subject_list),
                        ('side_id', side_list),
                        ('dataset_id', dataset_list)]

#for i in range(35):
#    if i < 10:
#        subject_list.append('train00' + str(i))
#    else:
#        subject_list.append('train0' + str(i))
#[0][0:26])
#i'm not so sure this will work now, we need to find a way of having all the subjects iterated over and the sides too.
#because this won't need to be iterated in every node, so we need to choose which nodes need to be iterated over for both left and right.

#infosource = Node(IdentityInterface(fields=['dataset', 'side']), name="infosource")
#infosource.iterables = [('dataset', iterable_list[0]), ('side', iterable_list[1])]
#because we are doing different things with the datasets now i think we should make different templates for the tses (mprage is all the same):

templates = {'umc_tse_native' : 'ashs_atlas_umcutrecht/train/{subject_id}/tse_native_chunk_{side_id}.nii.gz',
             'umc_tse_whole' : 'ashs_atlas_umcutrecht/train/{subject_id}/tse.nii.gz',
             'mag_tse_native' : 'ashs_atlas_magdeburg/train/{subject_id}/tse_native_chunk_{side_id}.nii.gz',
             'mag_tse_whole' : 'ashs_atlas_magdeburg/train/{subject_id}/tse.nii.gz',
             #seg
             'umc_seg_native' : 'ashs_atlas_umcutrecht/train/{subject_id}/tse_native_chunk_{side_id}_seg.nii.gz',
             'mag_seg_native' : 'ashs_atlas_magdeburg/train/{subject_id}/tse_native_chunk_{side_id}_seg.nii.gz',
             #mprage
             'umc_mprage_chunk' : 'ashs_atlas_umcutrecht/train/{subject_id}/mprage_to_chunktemp_{side_id}.nii.gz',
             'mag_mprage_chunk' : 'ashs_atlas_magdeburg/train/{subject_id}/mprage_to_chunktemp_{side_id}.nii.gz',
             }

bespoke_files = {'umc_tse_template' : 'umc_tse_template.nii.gz'}    

selectfiles = Node(SelectFiles(templates, base_directory=experiment_dir), name='selectfiles')

selecttemplates = Node(SelectFiles(bespoke_files, base_directory=github_dir), name='selecttemplates')

wf.connect([(infosource, selectfiles, [('subject_id', 'subject_id'),
                                       ('side_id', 'side_id'),
                                       ('dataset_id', 'dataset_id')])]) 



############
## Step 1 ##
############
#seeing as the UMC dataset is already close to isotropic, we will use it as our standard.
#Native chunks for TSE contain the segmentation, so we will keep them in this space.
#We need to change the whole tse to 0.35mm iso anyway to get the resolution consistent across all data
'''
# Maybe we don't need the out_files specified here?
umc_tse_resample_n = MapNode(C3d(interp = "Sinc", args = '-resample-mm 0.35x0.35x0.35mm', out_files = 'umc_tse_whole_resampled.nii.gz'),
                             name='umc_tse_resample_n', iterfield =['in_file']) 
wf.connect([(selectfiles, umc_tse_resample_n, [('umc_tse_whole','in_file')])])
'''

############
## Step 2 ##
############

#But the chunks are different sizes, so we will resize them to the correct size.
#pad the tse_native_chunk to the correct size, binarize, and resample

umc_tse_pad_bin_n = MapNode(C3d(interp = "Sinc", pix_type = 'float', args = '-resample-mm 0.35x0.35x0.35mm -pad-to 176x144x128 0 -binarize' , out_files = 'tse_chunk_resampled_padded_binarized.nii.gz'),
                            name='umc_tse_pad_bin_n', iterfield=['in_file']) 
wf.connect([(selectfiles, umc_tse_pad_bin_n, [('umc_tse_native','in_file')])])


## Step 2.5 resample+pad segmentation ##
umc_seg_resample_pad_n = MapNode(C3d(interp = "Sinc", pix_type = 'float', args = '-resample-mm 0.35x0.35x0.35mm -pad-to 176x144x128 0' , out_files = 'umc_chunk_seg_resampled_padded.nii.gz'),
                            name='umc_seg_resample_pad_n', iterfield=['in_file']) 
wf.connect([(selectfiles, umc_tse_pad_bin_n, [('umc_seg_native','in_file')])])


###### READY ###### Done with umc_native_seg and Ready for datasink
'''
umc_tse_pad_bin_n = MapNode(FLIRT(),
                            name='umc_tse_pad_bin_n', iterfield=['in_file']) 
wf.connect([(selectfiles, umc_tse_pad_bin_n, [('umc_tse_native','in_file')])])
wf.connect([(selectfiles, umc_tse_pad_bin_n, [('umc_tse_native','reference')])])
'''


############    
## Step 3 ##
############

#then multiply the bin mask by the original TSE to get the same sized chunks across the dataset. (prolly have to -reslice identity first
#because we have two inputs to multiply, we may need to add the output from the previous step to selectfiles?
# Or we can make the outfiles into variables? Not sure how to do this.
#


umc_tse_reslice_n =  MapNode(C3d(interp = "Sinc", pix_type = 'float', args = '-reslice-identity', out_files = 'umc_tse_chunk_resliced.nii.gz'),
                             name='umc_tse_reslice_n', iterfield =['in_file'])

wf.connect([(umc_tse_pad_bin_n, umc_tse_reslice_n, [('out_files','in_file') ])])

# For test purposes. --> Real line: wf.connect([(umc_tse_resample_n, umc_tse_reslice_n, [('out_files','opt_in_file')])])
wf.connect([(selectfiles, umc_tse_reslice_n, [('umc_tse_whole','opt_in_file')])])


###### READY ###### Done with umc_native_tse_chunk and Ready for datasink


'''
# should not be needed
#then multiply
umc_tse_mult_n = Node(C3d(interp = "Sinc", pix_type = 'float', args = '-multiply'),
                          name='umc_tse_mult_n') 
wf.connect([(umc_tse_reslice_n, umc_tse_mult_n, [('out_files','in_file')])])
wf.connect([(selectfiles, umc_tse_mult_n, [('umc_tse_whole','in_file')])])        
'''
############
## Step 4 ##
############
# The Mag data needs to be resampled to the right resolution, but doing this willblow the the z direction out in terms of image size
# So we register the TSEs of the magdeberg dataset to the template of the
# UMC dataset rigidly.The template is in the github repo cause we had to make it first (see bash script).
mag_register_n = MapNode(RegistrationSynQuick(transform_type = 'r', use_histogram_matching=True), 
                         name='mag_register_n', iterfield=['moving_image'])

wf.connect([(selecttemplates, mag_register_n, [('umc_tse_template', 'fixed_image')])])
wf.connect([(selectfiles, mag_register_n, [('mag_tse_whole', 'moving_image')])])


############
## Step 5 ##
############
#the mag data needs to have the same treatment as the umc data now.
#First, take the transformation that we just computed and apply it to the tse_native_chunk

mag_native_move_n = MapNode(ApplyTransforms(dimension = 3, interpolation = 'BSpline'),
                            name='mag_native_move_n', iterfield=['input_image', 'transforms', 'reference_image']) #Hmm, we should need to iterate over transforms as well??
wf.connect([(selectfiles, mag_native_move_n, [('mag_tse_native', 'input_image')])])
wf.connect([(mag_register_n, mag_native_move_n, [('out_matrix', 'transforms'),
                                                 ('warped_image','reference_image')])]) 

mag_seg_native_move_n = MapNode(ApplyTransforms(dimension = 3, interpolation = 'BSpline'),
                            name='mag_seg_native_move_n', iterfield=['input_image', 'transforms', 'reference_image'])
wf.connect([(selectfiles, mag_seg_native_move_n, [('mag_seg_native', 'input_image')])])
wf.connect([(mag_register_n, mag_seg_native_move_n, [('out_matrix', 'transforms'),
                                                 ('warped_image','reference_image')])])




##################
##  New Step 6  ##
##################
# Repeat Steps 2-3 but with registered Magdeburg

# Do we need resampling here after applying transformations further up??
mag_tse_pad_bin_n = MapNode(C3d(interp = "Sinc", pix_type = 'float', args = '-resample-mm 0.35x0.35x0.35mm -pad-to 176x144x128 0 -binarize' , out_files = 'mag_chunk_resampled_padded_binarized.nii.gz'),
                            name='mag_tse_pad_bin_n', iterfield=['in_file']) 
wf.connect([(mag_native_move_n, mag_tse_pad_bin_n, [('output_image','in_file')])])

# resample+pad mag_seg_native here?? 
###### READY ###### Then ready for Datasink!


mag_tse_reslice_n =  MapNode(C3d(interp = "Sinc", pix_type = 'float', args = '-reslice-identity', out_files = 'mag_tse_chunk_resliced.nii.gz'),
                             name='mag_tse_reslice_n', iterfield =['in_file'])

wf.connect([(mag_tse_pad_bin_n, mag_tse_reslice_n, [('out_files','in_file') ])])

# NOTE!! # For test purposes. --> Real line: wf.connect([(mag_register_n, mag_tse_reslice_n, [('warped_image','opt_in_file')])])
wf.connect([(selectfiles, mag_tse_reslice_n, [('mag_tse_whole','opt_in_file')])])

###### READY ###### mag_tse_native_chunk ready for Datasink



##################
##  New Step 7  ##
##################
# Ants the mprage to the freshly cutout tse_native_chunks
# Currently this doesn't seem to work properly as the images does not have enough mutual information
# Additionally it hangs for more than an hour, so still an unfinished node here.

## Magdeburg ##
mag_mprage_to_tse_register_n = MapNode(RegistrationSynQuick(), # Tom has no registration input parameters here 
                         name='mag_mprage_to_tse_register_n', iterfield=['moving_image'])

## Register to non padded warped mag_tse_chunk?
wf.connect([(mag_native_move_n, mag_mprage_to_tse_register_n, [('output_image', 'fixed_image')])])
wf.connect([(selectfiles, mag_mprage_to_tse_register_n, [('mag_mprage_chunk', 'moving_image')])])


## UMC ##
umc_mprage_to_tse_register_n = MapNode(RegistrationSynQuick(), # Tom has no registration input parameters here 
                         name='umc_mprage_to_tse_register_n', iterfield=['moving_image'])

## Register to resampled or resliced mag_tse_chunk? We don't have just the resampled one currently
# Tom currently registers to the whole resampled image, hmm... 
wf.connect([(umc_tse_reslice_n, umc_mprage_to_tse_register_n, [('out_files', 'fixed_image')])])
wf.connect([(selectfiles, umc_mprage_to_tse_register_n, [('umc_mprage_chunk', 'moving_image')])])


# Remember to sink your data like a good boy

datasink = Node(DataSink(base_directory=experiment_dir+working_dir,
                         container=output_dir),
                name="datasink")

wf.connect([(umc_mprage_to_tse_register_n, datasink, [('warped_image', 'umc_mprage_to_tse_register')])])

wf.run()



'''
##############
##  Step 4  ##
##############

#normalise_mprage_n = MapNode(C3d(histmatch=True),
#                             name='normalise_mprage_n', iterfield=['in_file'])
#wf.connect([(selectfiles, mprage_flirt_n, [('out_file', 'in_file')])]) #check that this works FIXME
#include the reference mprage here
#normalise_tse_n = MapNode(C3d(histmatch=True),
#                          name='normalise_tse_n', iterfield=['in_file'])
#wf.connect([(selectfiles, tse_flirt_n, [('out_file', 'in_file')])])
#wf.connect #need to include the reference file here







################
## DATA SINK  ##
################
datasink = Node(DataSink(base_directory=experiment_dir,
                         container=output_dir),
                name="datasink")

wf.connect([(umc_tse_mult_n, datasink, [('out_files', 'umc_native_resized_final')])])
#wf.connect([(n, datasink, [('out_file', 'tse_resized')])])
#wf.connect([(n, datasink, [('out_file', 'segmentation_resized')])])
#wf.connect([(n, datasink, [('output_image', 'mprage_resized_normalised')])])
#wf.connect([(n, datasink, [('output_image', 'tse_resized_normalised')])])


###################
## Run the thing ##
###################
# # run as MultiProc
wf.write_graph(graph2use='flat', format='png', simple_form=False)
#wf.run(plugin='SLURMGraph', plugin_args = {'dont_resubmit_completed_jobs': True} )

wf.run(plugin='MultiProc', plugin_args={'n_procs' : 20})
'''
#This is for running at CAI
# qsub_args='-N 1,-c 4,--partition=all, --mem=16000'))
#running at Awoonga:
