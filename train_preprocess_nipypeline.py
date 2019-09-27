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
experiment_dir = '/RDS/Q1219/data/' 
output_dir = '/RDS/Q1219/data/preprocessed'
working_dir = '/30days/${USER}/DEEPSEACAT_WORKINGDIR'
github_dir = '~/DEEPSEACAT-Deep-learning-for-Segmentation-And-CharActerisation-of-Tissue/ '
################

dataset = ['magdeburg', 'umcutrecht']
side = ['left', 'right']

subject_list = []
for i in range(35):
    if i < 10:
        subject_list.append('train00' + str(i))
    else:
        subject_list.append('train0' + str(i))

iterable_list = [dataset, subject_list, side]
#iterable_list = [dataset, side]

wf = Workflow(name='train_preprocess_DL_hippo') 
wf.base_dir = os.path.join(experiment_dir, 'working_dir
')

# create infosource to iterate over iterables
infosource = Node(IdentityInterface(fields=['dataset', 'subject_list', 'side']), name="infosource")
infosource.iterables = [('dataset', iterable_list[0]),('subject_list', iterable_list[1][0:26]), ('side', iterable_list[2])]

#i'm not so sure this will work now, we need to find a way of having all the subjects iterated over and the sides too.


#infosource = Node(IdentityInterface(fields=['dataset', 'side']), name="infosource")
#infosource.iterables = [('dataset', iterable_list[0]), ('side', iterable_list[1])]


#because we are doing different things with the datasets now i think we should make different templates for the tses (mprage is all the same):

templates = {'umc_tse_native': 'ashs_atlas_umcutrecht_7t_20170810/train/{subject_list}/tse_native_chunk_{side}.nii.gz'
             'umc_tse_whole': 'ashs_atlas_umcutrecht_7t_20170810/train/{subject_list}/tse.nii.gz'
             'mag_tse_native': 'ashs_atlas_magdeburg_7t_20180416/train/{subject_list}/tse_native_chunk_{side}.nii.gz'
             'mag_tse_whole': 'ashs_atlas_magdeburg_7t_20180416/train/{subject_list}/tse.nii.gz'
             #seg
             'umc_seg_native': 'ashs_atlas_umcutrecht_7t_20170810/train/{subject_list}/tse_native_chunk_{side}_seg.nii.gz'
             'mag_seg_native': 'ashs_atlas_magdeburg_7t_20180416/train/{subject_list}/tse_native_chunk_{side}_seg.nii.gz'
             #mprage
             'mprage_chunk': 'ashs_atlas_{dataset}/train/{subject_list}/mprage_to_chunktemp_{side}.nii.gz'}

bespoke_files = {'umc_tse_template' : 'lib/umc_tse_template.nii.gz'}    

selectfiles = Node(SelectFiles(templates, base_directory=experiment_dir), name='selectfiles')
selecttemplates = Node(SelectFiles(bespoke_files, base_dir=github_dir), name='selecttemplates')

wf.connect([(infosource, selectfiles, [('subject_id', 'subject_id')])]) 

#templates = {'seg_whole-image':  'ashs_atlas_{dataset}/train/train*/seg_{side}.nii.gz',
#             'mprage_chunk':     'ashs_atlas_{dataset}/train/train*/mprage_to_chunktemp_{side}.nii.gz',
#             'tse_whole-image':  'ashs_atlas_{dataset}/train/train*/tse.nii.gz'}


#mprage_flirt_n = MapNode(FLIRT(uses_qform=True, apply_xfm=True, applyisoxfm=0.35, interp='sinc', datatype='float'),
#                         name='mprage_flirt_n', iterfield=['in_file']) # iterfield forventer et input som er en liste med inputnavne og ikke kun et navn, vi skal altsaa have en liste der indeholder alle mprage billeder for begge datasaet og begge sider
#wf.connect([(selectfiles, mprage_flirt_n, [('mprage_to_chunktemp_{side}.nii','template.nii')])]) #Skriv navnet paa in_file (inputlisten) og referencebilledet (templaten)

#wf.connect([(selectfiles, mprage_flirt_n, [('{side}_template_mprage_chunk', 'reference')])])train*
#wf.connect([(selectfiles, mprage_flirt_n, [('{side}_mprage_chunk', 'in_file')])])
#wf.connect([(selectfiles, mprage_flirt_n, [('mprage_to_chunktemp_{side}.nii.gz', 'in_file')])])
#wf.connect([(selectfiles, flirt_n, [('', 'out_file')])])


############
## Step 1 ##
############
#seeing as the UMC dataset is already close to isotropic, we will use it as our standard.
#Native chunks for TSE contain the segmentation, so we will keep them in this space.
#We need to change the whole tse to 0.35mm iso anyway to get the resolution consistent across all data

umc_tse_resample_n = MapNode(C3d(interp = "Sinc", pix_type = 'float', args='resample-mm 0.35x0.35x0.35mm'),
                             name='umc_tse_resample_n', iterfield=['in_file']) 
wf.connect([(selectfiles, umc_tse_resample_n, [('umc_tse_whole','in_file')])])

############
## Step 2 ##
############

#But the chunks are different sizes, so we will resize them to the correct size.
#pad the tse_native_chunk to the correct size, binarize, and resample

umc_tse_pad_bin_n = MapNode(C3d(interp = "Sinc", pix_type = 'float', args='resample-mm 0.35x0.35x0.35mm', args='pad-to 176x144x128 0', args='-binarize'),
                            name='umc_tse_pad_bin_n', iterfield=['in_file']) 
wf.connect([(selectfiles, umc_tse_pad_bin_n, [('umc_tse_chunk','in_file')])])

############
## Step 3 ##
############
#then multiply the bin mask by the original TSE to get the same sized chunks across the dataset. (prolly have to -reslice identity first


umc_tse_reslice_n =  MapNode(C3d(interp = "Sinc", pix_type = 'float', args='-reslice-identity'),
                             name='umc_tse_reslice_n', iterfield=['in_file']) 
wf.connect([(umc_tse_pad_bin_n, umc_tse_reslice_n, [('out_file','in_file')])])
wf.connect([(selectfiles, umc_tse_resclice_n, [('umc_tse_whole','in_file')])])

#then multiply
umc_tse_mult_n =  MapNode(C3d(interp = "Sinc", pix_type = 'float', args='-multiply'),
                          name='umc_tse_mult_n', iterfield=['in_file']) 
wf.connect([(umc_tse_resclice_n, umc_tse_mult_n, [('out_file','in_file')])])
wf.connect([(selectfiles, umc_tse_mult_n, [('umc_tse_whole','in_file')])])        

############
## Step 4 ##
############
# The Mag data needs to be resampled to the right resolution, but doing this willblow the the z direction out in terms of image size
# So we register the TSEs of the magdeberg dataset to the template of the
#UMC dataset rigidly.The template is in the github repo cause we had to make it first (see bash script).
mag_register_n = MapNode(RegistrationSynQuick(transform_type = 'r', use_histogram_matching=True ), 
                         name='mag_register_n', iterfield=['moving_image'])
wf.connect([(selecttemplates, mag_register_n, [('umc_tse_template', 'fixed_image')])])
wf.connect([(selectfiles, mag_register_n, [('mag_tse_whole', 'moving_image')])])

############
## Step 5 ##
############
#the mag data needs to have the same treatment as the umc data now.
#First, take the transformation that we just computed and apply it to the tse_native_chunk

mag_native_move_n = MapNode(ApplyTransforms(dimension = '3', interpolation = 'BSpline', 
                                            name='mag_native_move_n', iterfield=['input_file'])
                            wf.connect([(selectfiles, mag_native_move_n, [('mag_tse_native', 'input_file')])])
                            wf.connect([(mag_register_n, mag_native_move_n, [('', 'transforms')])]) #need to figure out where the affine is from the previous step. Does it have a standard name?

    
######got up to here.





             ######################
             ##  Step 3  ##
             #Resample seg#
             ######################

segmentation_n = MapNode(FLIRT(uses_qform=True, apply_xfm=True, applyisoxfm=0.35, interp='nearestneighbour', datatype='float'), 
                         name='segment_flirt_n', iterfield=['in_file'])
wf.connect([(selectfiles, mprage_flirt_n, [('out_file', 'reference')])]) #check that this works FIXME
wf.connect([(selectfiles, segmentation_n, [('seg', 'in_file')])])

##############
##  Step 4  ##
##############

normalise_mprage_n = MapNode(C3d(histmatch=True),
                             name='normalise_mprage_n', iterfield=['in_file'])
wf.connect([(selectfiles, mprage_flirt_n, [('out_file', 'in_file')])]) #check that this works FIXME
#include the reference mprage here
normalise_tse_n = MapNode(C3d(histmatch=True),
                          name='normalise_tse_n', iterfield=['in_file'])
wf.connect([(selectfiles, tse_flirt_n, [('out_file', 'in_file')])])
wf.connect #need to include the reference file here


################
## DATA SINK  ##
################
datasink = Node(DataSink(base_directory=experiment_dir, container=output_dir),
                name='datasink')
wf.connect([(mprage_flirt_n, datasink, [('out_file', 'mprage_resized')])])
wf.connect([(tse_flirt_n, datasink, [('out_file', 'tse_resized')])])
wf.connect([(segmentation_n, datasink, [('out_file', 'segmentation_resized')])])
wf.connect([(normalise_mprage_n, datasink, [('output_image', 'mprage_resized_normalised')])])
wf.connect([(normalise_tse_n, datasink, [('output_image', 'tse_resized_normalised')])])


###################
## Run the thing ##
###################
# # run as MultiProc
wf.write_graph(graph2use='flat', format='png', simple_form=False)
#wf.run('MultiProc', plugin_args={'n_procs': 20})
#This is for running at CAI
wf.run(plugin='SLURMGraph', plugin_args=dict(
    qsub_args='-N 1,-c 4,--partition=long,wks,all, --mem=16000'))
