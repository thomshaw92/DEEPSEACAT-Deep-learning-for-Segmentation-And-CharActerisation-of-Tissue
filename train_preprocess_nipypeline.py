#!/usr/bin/env python3
from os.path import join as opj
import os
from nipype.interfaces import fsl
from nipype.interfaces.base import (TraitedSpec, 
	CommandLineInputSpec, 
	CommandLine, 
	File, 
	traits
)
#from nipype.interfaces.ants import N4BiasFieldCorrection
#from nipype.interfaces.ants.segmentation import BrainExtraction
from nipype.interfaces.fsl import flirt
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.pipeline.engine import Workflow, Node, MapNode
#from nipype.interfaces.ants.segmentation import CorticalThickness

#test git
# from nipype import config
# config.enable_debug_mode()
#
# config.set('execution', 'stop_on_first_crash', 'true')
# config.set('execution', 'remove_unnecessary_outputs', 'false')
# config.set('execution', 'keep_inputs', 'true')
# config.set('logging', 'workflow_level', 'DEBUG')
# config.set('logging', 'interface_level', 'DEBUG')
# config.set('logging', 'utils_level', 'DEBUG')

os.environ["FSLOUTPUTTYPE"] = "NIFTI_GZ"

# work on scratch space only
experiment_dir = '/ashs_atlas_umcutrecht_7t_20170810/train/'
output_dir = '/RDM'
working_dir = '/scratch'

subject_list = ['train000' 'etc']
#session_list = ['ses-01']

iterable_list = [subject_list, session_list]

wf = Workflow(name='train_preprocess_DL_hippo')
wf.base_dir = opj(experiment_dir, working_dir)

# create infosource to iterate over subject list
infosource = Node(IdentityInterface(fields=['subject_id', 'session_id']), name="infosource")
infosource.iterables = [('subject_id', subject_list)]

infosource = Node(IdentityInterface(fields=['subject_id']), name="infosource")
infosource.iterables = [('subject_id')]

templates = {'seg': '{subject_id}/seg_left.nii.gz',
             'mprage_chunk': '{subject_id}/anat/*T1w*.nii.gz',
             'tse': '{subject_id}/anat/.nii.gz',
}
selectfiles = Node(SelectFiles(templates, base_directory=experiment_dir), name='selectfiles')

#PLAN:# create a synthetic (fake) input image size and res of 176x144x128 and 0.35 mm iso
# Cut the TSE first(mul MP2RAGE) using flirt and resample to the synth data.
# resample everything  else into that space and siz too.
#normalise all using ImageMaths 
#pad (not yet)
                                       
wf.connect([(infosource, selectfiles, [('subject_id', 'subject_id')])])
###########
## synth data 
##############

#resample an image to the correct size and res and then multiply by 0
fslmaths input.nii -bin out.nii

###
fake_n = MapNode


####FSLMATHS##
m###ultiply TSE by synthetic
###fslmaths tse.nii -mul synth.nii tse_chunk_correct_size.nii

###########
## flirt ##
###########

flirt_n = MapNode(fsl.FLIRT(uses_qform=True, apply_xfm=True
                  name='flirt_n', iterfield=['in_file'])
wf.connect([(selectfiles, flirt_n, [('tse', 'reference')])])
wf.connect([(selectfiles, flirt_n_flair, [('flair', 'in_file')])])
wf.connect([(selectfiles, flirt_n_flair, [('t1w', 'out_file')])])








####################
## ants_brain_ext ##
####################
ants_be_n = MapNode(BrainExtraction(dimension=3, brain_template='/data/fasttemp/uqtshaw/tomcat/data/derivatives/myelin_mapping/T_template.nii.gz', brain_probability_mask='/data/fasttemp/uqtshaw/tomcat/data/derivatives/myelin_mapping/T_template_BrainCerebellumProbabilityMask.nii.gz'),
		name='ants_be_node', iterfield=['anatomical_image'])
wf.connect([(selectfiles, ants_be_n, [('t1w', 'anatomical_image')])]) 
############
## antsCT ##
############
antsct_n = MapNode(CorticalThickness(dimension=3, brain_template='/data/fastertemp/uqtshaw/ANTS/ants_ct_templates/T_template.nii.gz', brain_probability_mask='/data/fastertemp/uqtshaw/ANTS/ants_ct_templates/T_template_BrainCerebellumProbabilityMask.nii.gz', segmentation_priors=['/data/fastertemp/uqtshaw/ANTS/ants_ct_templates/Priors/priors1.nii.gz', '/data/fastertemp/uqtshaw/ANTS/ants_ct_templates/Priors/priors2.nii.gz', '/data/fastertemp/uqtshaw/ANTS/ants_ct_templates/Priors/priors3.nii.gz', '/data/fastertemp/uqtshaw/ANTS/ants_ct_templates/Priors/priors4.nii.gz', '/data/fastertemp/uqtshaw/ANTS/ants_ct_templates/Priors/priors5.nii.gz'], t1_registration_template='/data/fastertemp/uqtshaw/ANTS/ants_ct_templates/T_template_BrainCerebellum.nii.gz'),
              name='ants_node', iterfield=['anatomical_image'])
wf.connect([(selectfiles, antsct_n, [('t1w', 'anatomical_image')])]) 
###############
## mult_mask ##
###############
mult_mask_n_flair = MapNode(ImageMaths(op_string='-mul'),
                      name="mult_mask_flair", iterfield=['in_file', 'in_file2'])
wf.connect([(ants_be_n, mult_mask_n_flair, [('BrainExtractionMask', 'in_file')])])
wf.connect([(flirt_n_flair, mult_mask_n_flair, [('out_file', 'in_file2')])])
mult_mask_n_space = MapNode(ImageMaths(op_string='-mul'),
                      name="mult_mask_space", iterfield=['in_file', 'in_file2'])
wf.connect([(ants_be_n, mult_mask_n_space, [('BrainExtractionMask', 'in_file')])])
wf.connect([(flirt_n_space, mult_mask_n_space, [('out_file', 'in_file2')])])
###############
## N4 the T2 ##
###############
n4_n_flair = MapNode(N4BiasFieldCorrection(dimension=3, bspline_fitting_distance=300, shrink_factor=3, n_iterations=[50,50,30,20]),
                        name="n4_flair", iterfield=['input_image'])
wf.connect([(mult_mask_n_flair, n4_n_flair, [('out_file', 'input_image')])])
n4_n_space = MapNode(N4BiasFieldCorrection(dimension=3, bspline_fitting_distance=300, shrink_factor=3, n_iterations=[50,50,30,20]),
                        name="n4_space", iterfield=['input_image'])
wf.connect([(mult_mask_n_space, n4_n_space, [('out_file', 'input_image')])])


################
## DATA SINK  ##
################
datasink = Node(DataSink(base_directory=experiment_dir, container=output_dir),
                name='datasink')
wf.connect([(mult_mask_n_space, datasink, [('out_file', 'spacemasked')])])
wf.connect([(mult_mask_n_flair, datasink, [('out_file', 'flairmasked')])])
wf.connect([(n4_n_space, datasink, [('output_image', 'spaceN4')])])
wf.connect([(n4_n_flair, datasink, [('output_image', 'flairN4')])])
wf.connect([(antsct_n, datasink, [('BrainSegmentation', 'brainsegmentation')])])
wf.connect([(ants_be_n, datasink, [('BrainExtractionMask', 'mask_ants_t1w')])])
wf.connect([(ants_be_n, datasink, [('BrainExtractionBrain', 'brain_ants_t1w')])])

# # run as MultiProc
wf.write_graph(graph2use='flat', format='png', simple_form=False)


#wf.run('MultiProc', plugin_args={'n_procs': 20})

wf.run(plugin='SLURMGraph', plugin_args=dict(
    qsub_args='-N 1,-c 4,--partition=long,wks,all, --mem=16000'))


#robex for future ref
class RobexInputSpec(CommandLineInputSpec):
    in_file = File(desc = "t1w", exists = True,
         mandatory = True, position = 0, argstr="%s")
    out_file = File(desc = "Output volume", position = 1, argstr="%s", name_source=['in_file'],
                        hash_files=False, name_template='%s_brain', keep_extension=True)
    out_mask = File(desc = "Output mask", position = 2, argstr="%s", name_source=['in_file'],
                        hash_files=False, name_template='%s_brainmask', keep_extension=True)
    seed = traits.Int(desc = "seed for random number generator", position = 3, argstr = "%i")
class RobexOutputSpec(TraitedSpec):
    out_file = File(desc = "Output volume", exists = True)
    out_mask = File(desc = "Output mask", exists = True)
class RobexTask(CommandLine):
    input_spec = RobexInputSpec
    output_spec = RobexOutputSpec
    _cmd = 'runROBEX.sh'
