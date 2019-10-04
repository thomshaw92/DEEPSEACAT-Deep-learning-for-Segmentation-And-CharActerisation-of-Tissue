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
experiment_dir = '/winmounts/uqmtottr/uq-research/DEEPSEACAT-Q1219/data/'
#where all the atlases live
#github_dir = '/data/home/uqmtottr/DEEPSEACAT-Deep-learning-for-Segmentation-And-CharActerisation-of-Tissue/lib/'
github_dir = '/winmounts/uqmtottr/uq-research/DEEPSEACAT-Q1219/data/DEEPSEACAT_atlas/'
##############
#the outdir
output_dir = 'output_dir'
#working_dir name
working_dir = 'Mette_Nipype_working_dir'
#other things to be set up
side_list = ['left', 'right']
subject_list = ['train000', 'train001']
#subject_list = sorted(os.listdir(experiment_dir+'ashs_atlas_magdeburg/train/'))
#shorter_list = sorted(os.listdir(experiment_dir+'ashs_atlas_umcutrecht/train/')) 
#####################

wf = Workflow(name='Workflow_preprocess_DL_hippo') 
wf.base_dir = os.path.join(experiment_dir+working_dir)

# create infosource to iterate over iterables
infosource = Node(IdentityInterface(fields=['subject_id',
                                            'side_id',
                                            'dataset_id']),
                  name="infosource")
infosource.iterables = [('subject_id', subject_list),
                        #('shorter_id', shorter_list),
                        ('side_id', side_list)]


templates = {#tse
             'umc_tse_native' : 'ashs_atlas_umcutrecht/train/{subject_id}/tse_native_chunk_{side_id}.nii.gz',
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
# change and add more strings to include all necessary templates
bespoke_files = {'umc_tse_whole_template' : 'umc_tse_template_resampled_0.35mm.nii.gz'}    

selectfiles = Node(SelectFiles(templates, base_directory=experiment_dir), name='selectfiles')

selecttemplates = Node(SelectFiles(bespoke_files, base_directory=github_dir), name='selecttemplates')

wf.connect([(infosource, selectfiles, [('subject_id', 'subject_id'),
                                       #('shorter_id', 'shorter_id'),
                                       ('side_id', 'side_id')])]) 

############
## Step 1 ##
############
#seeing as the UMC dataset is already close to isotropic, we will use it as our standard.
#Native chunks for TSE contain the segmentation, so we will keep them in this space.
#We need to change the whole tse to 0.35mm iso anyway to get the resolution consistent across all data
# Maybe we don't need the out_files specified here?
umc_tse_whole_resample_n = MapNode(C3d(interp = "Sinc", args = '-resample-mm 0.35x0.35x0.35mm', out_files = 'umc_tse_whole_resampled.nii.gz'),
                             name='umc_tse_whole_resample_n', iterfield =['in_file']) 
wf.connect([(selectfiles, umc_tse_whole_resample_n, [('umc_tse_whole','in_file')])])
# It seems that the resampling doesn't work to exactly 0.35 iso, so maybe this step is unnessecary or we should try with 0.3500mm iso in the code?

############
## Step 2 ##
############
#But the chunks are different sizes, so we will resize them to the correct size.
#pad the tse_native_chunk to the correct size, binarize, and resample
umc_tse_resampled_pad_bin_n = MapNode(C3d(interp = "Sinc", pix_type = 'float', args = '-resample-mm 0.35x0.35x0.35mm -pad-to 176x144x128 0 -binarize' , out_files = 'umc_tse_chunk_resampled_padded_binarized.nii.gz'),
                            name='umc_tse_resampled_pad_bin_n', iterfield=['in_file']) 
wf.connect([(selectfiles, umc_tse_resampled_pad_bin_n, [('umc_tse_native','in_file')])])


###############
## Step 3 ##
###############
# Resample and pad the segmentation chunk from umc - tse_native_chunk_seg
# Do as step 2 without the binarization
umc_seg_resample_pad_n = MapNode(C3d(interp = "Sinc", pix_type = 'float', args = '-resample-mm 0.35x0.35x0.35mm -pad-to 176x144x128 0', out_files = 'umc_chunk_seg_resampled_padded.nii.gz'),
                                     name='umc_seg_resample_pad_n', iterfield=['in_file'])
wf.connect([(selectfiles, umc_seg_resample_pad_n, [('umc_seg_native','in_file')])])


############    
## Step 4 ##
############
#then reslice the bin mask by the original TSE to get new same sized chunks across the dataset.
umc_tse_reslice_n =  MapNode(C3d(interp = "Sinc", pix_type = 'float', args = '-reslice-identity', out_files = 'umc_new_tse_chunk_resliced.nii.gz'),
                             name='umc_tse_reslice_n', iterfield =['in_file'])

wf.connect([(umc_tse_resampled_pad_bin_n, umc_tse_reslice_n, [('out_files','in_file') ])])
wf.connect([(umc_tse_whole_resample_n, umc_tse_reslice_n, [('out_files','opt_in_file')])])


############
## Step 5 ##
############
# The Mag data needs to be resampled to the right resolution, but doing this will blow the z direction out in terms of image size
# So we register the TSEs of the magdeberg dataset to the template of the UMC dataset rigidly.
# The template is in the github repo cause we had to make it first (see bash script).
mag_register_n = MapNode(RegistrationSynQuick(transform_type = 'r', use_histogram_matching=True), 
                         name='mag_register_n', iterfield=['moving_image'])
wf.connect([(selecttemplates, mag_register_n, [('umc_tse_whole_template', 'fixed_image')])])
wf.connect([(selectfiles, mag_register_n, [('mag_tse_whole', 'moving_image')])])


############
## Step 6 ##
############
#the mag data needs to have the same treatment as the umc data now.
#First, take the transformation that we just computed and apply it to the mag_tse_native_chunk
mag_native_move_n = MapNode(ApplyTransforms(dimension = 3, interpolation = 'BSpline'),
                            name='mag_native_move_n', iterfield=['input_image', 'transforms', 'reference_image']) #Hmm, we should need to iterate over transforms as well??
wf.connect([(selectfiles, mag_native_move_n, [('mag_tse_native', 'input_image')])])
wf.connect([(mag_register_n, mag_native_move_n, [('out_matrix', 'transforms'),
                                                 ('warped_image','reference_image')])]) 

# then apply the transformation to the mag_tse_native_seg
mag_seg_native_move_n = MapNode(ApplyTransforms(dimension = 3, interpolation = 'BSpline'),
                            name='mag_seg_native_move_n', iterfield=['input_image', 'transforms', 'reference_image'])
wf.connect([(selectfiles, mag_seg_native_move_n, [('mag_seg_native', 'input_image')])])
wf.connect([(mag_register_n, mag_seg_native_move_n, [('out_matrix', 'transforms'),
                                                 ('warped_image','reference_image')])])


##################
##  New Step 7  ##
##################
# Repeat Steps 2-4 but with registered Magdeburg
#First we pad and binarizes the mag tse native chunk
mag_tse_pad_bin_n = MapNode(C3d(interp = "Sinc", pix_type = 'float', args = '-pad-to 176x144x128 0 -binarize' , out_files = 'mag_tse_chunk_padded_binarized.nii.gz'),
                            name='mag_tse_pad_bin_n', iterfield=['in_file']) 
wf.connect([(mag_native_move_n, mag_tse_pad_bin_n, [('output_image','in_file')])])

# Second we resample and pad the mag_seg_native 
mag_seg_resample_pad_n = MapNode(C3d(interp = "Sinc", pix_type = 'float', args = '-resample-mm 0.35x0.35x0.35mm -pad-to 176x144x128 0', out_files = 'mag_chunk_seg_resampled_padded.nii.gz'),
                                     name='mag_seg_resample_pad_n', iterfield=['in_file'])
wf.connect([(mag_seg_native_move_n, mag_seg_resample_pad_n, [('output_image','in_file')])])

#Third the reslice the whole mag tse image to a new chunk based on the mag_tse_resample_pad_bin
mag_tse_reslice_n =  MapNode(C3d(interp = "Sinc", pix_type = 'float', args = '-reslice-identity', out_files = 'mag_new_tse_chunk_resliced.nii.gz'),
                             name='mag_tse_reslice_n', iterfield =['in_file'])

wf.connect([(mag_tse_pad_bin_n, mag_tse_reslice_n, [('out_files','in_file') ])])
wf.connect([(mag_register_n, mag_tse_reslice_n, [('warped_image','opt_in_file')])])

##################
##  New Step 8  ##
##################
# Ants the mprage to the freshly cutout tse_native_chunks
# Currently this doesn't seem to work properly as the images does not have enough mutual information
# Additionally it hangs for more than an hour.

## Magdeburg ##
mag_mprage_to_tse_register_n = MapNode(RegistrationSynQuick(), # Tom has no registration input parameters here 
                         name='mag_mprage_to_tse_register_n', iterfield=['moving_image'])

## Register to non padded warped mag_tse_chunk?
wf.connect([(mag_native_move_n, mag_mprage_to_tse_register_n, [('output_image', 'fixed_image')])])
wf.connect([(selectfiles, mag_mprage_to_tse_register_n, [('mag_mprage_chunk', 'moving_image')])])

### OBS Here we have to pad the mprage image to get the correct size

## UMC ##
umc_mprage_to_tse_register_n = MapNode(RegistrationSynQuick(), # Tom has no registration input parameters here 
                         name='umc_mprage_to_tse_register_n', iterfield=['moving_image'])

## Register to resampled or resliced mag_tse_chunk? We don't have just the resampled one currently
# Tom currently registers to the whole resampled image, hmm... 
wf.connect([(umc_tse_reslice_n, umc_mprage_to_tse_register_n, [('out_files', 'fixed_image')])])
wf.connect([(selectfiles, umc_mprage_to_tse_register_n, [('umc_mprage_chunk', 'moving_image')])])

### OBS Here we have to pad the mprage image to get the correct size

'''
##############
##  Step 9  ##
##############
#Normalization of the images to the templates for the overall tse intensity histogram and the overall mprage intensity histogram 
#mprage normalization
normalise_mprage_n = MapNode(C3d(interp="Sinc", pix_type='float', args='-histmatch 5'),
                             name='normalise_mprage_n', iterfield=['in_file'])
#wf.connect([(selectfiles, mprage_flirt_n, [('out_file', 'in_file')])]) #check that this works FIXME
#include the reference mprage here

#tse normalization
#normalise_tse_n = MapNode(C3d(interp="Sinc", pix_type='float', args='-histmatch 5'),
#                          name='normalise_tse_n', iterfield=['in_file'])
#wf.connect([(selectfiles, tse_flirt_n, [('out_file', 'in_file')])])
#wf.connect #need to include the reference file here
'''

################
## DATA SINK  ##
################
datasink = Node(DataSink(base_directory=experiment_dir+working_dir,
                         container=output_dir),
                name="datasink")
wf.connect([(umc_tse_whole_resample_n, datasink, [('out_files','umc_tse_whole_resample')])]) #Step 1
wf.connect([(umc_tse_resampled_pad_bin_n, datasink, [('out_files','umc_tse_resample_pad_bin')])]) #Step 2
wf.connect([(umc_seg_resample_pad_n, datasink, [('out_files', 'umc_seg_resample_pad')])]) #Step 3
wf.connect([(umc_tse_reslice_n, datasink, [('out_files','umc_new_tse_chunk_reslice')])]) #Step 4
wf.connect([(mag_register_n, datasink, [('warped_image','mag_tse_register_to_umc_tse_template')])]) #Step 5
wf.connect([(mag_native_move_n, datasink, [('output_image','mag_tse_native_transform')])]) #Step 6
wf.connect([(mag_seg_native_move_n, datasink, [('output_image','mag_tse_seg_native_transform')])]) #Step 6
wf.connect([(mag_tse_pad_bin_n, datasink, [('out_files','mag_tse_pad_bin')])]) #Step 7
wf.connect([(mag_seg_resample_pad_n, datasink, [('out_files','mag_seg_resample_pad')])]) #Step 7
wf.connect([(mag_tse_reslice_n, datasink, [('out_files','mag_new_tse_chunk_reslice')])]) #Step 7
wf.connect([(mag_mprage_to_tse_register_n, datasink [('warped_image','mag_mprage_to_tse_register')])]) #Step 8
#wf.connect for the mag mprage padding ... #Step 8
wf.connect([(umc_mprage_to_tse_register_n, datasink [('warped_image','umc_mprage_to_tse_register')])]) #Step 8
#wf.connect for the umc mprage padding ... #Step 8
#wf.connect([(normalise_mprage_n, datasink  [('out_files','')])]) #Step 9
#wf.connect([(normalise_tse_n, datasink [('out_files', '')])]) #Step 9
wf.run()
'''

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
