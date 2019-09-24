#!/usr/bin/env python3
from os.path import join as opj
import os
from nipype.interfaces.base import (TraitedSpec,
	CommandLineInputSpec, 
	CommandLine, 
	File, 
	traits
)
#from nipype.interfaces.ants import N4BiasFieldCorrection
#from nipype.interfaces import fsl
from nipype.interfaces.c3 import C3d
from nipype.interfaces.fsl import flirt
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.pipeline.engine import Workflow, Node, MapNode

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
# work on scratch space only
experiment_dir = '/ashs_atlas_umcutrecht_7t_20170810/' #we need to re-curate this data into a new directory with both atlasses.
output_dir = '/RDM'
working_dir = '/scratch'



dataset = ['magdeburg', 'umcutrecht']
side = ['left', 'right']
subject_list = []
for i in range(35):
    if i < 10:
        subject_list.append('train00' + str(i))
    else:
        subject_list.append('train0' + str(i))

iterable_list = [dataset, subject_list, side]

wf = Workflow(name='train_preprocess_DL_hippo') 
wf.base_dir = os.path.join(experiment_dir, 'out_preproces')

# create infosource to iterate over iterables
infosource = Node(IdentityInterface(fields=['dataset', 'subject_list', 'side']), name="infosource")
infosource.iterables = [('dataset', iterable_list[0]),('subject_list', iterable_list[1][0:26]), ('side', iterable_list[2])]


templates = {'seg_whole-image':  'ashs_atlas_{dataset}/train/{subject_list}/seg_{side}.nii.gz',
             'mprage_chunk':     'ashs_atlas_{dataset}/train/{subject_list}/mprage_to_chunktemp_{side}.nii.gz',
             'tse_whole-image':  'ashs_atlas_{dataset}/train/{subject_list}/tse.nii.gz',
}


selectfiles = Node(SelectFiles(templates, base_directory=experiment_dir), name='selectfiles')


wf.connect([(infosource, selectfiles, 
             [
              ('dataset', 'dataset'), 
              ('subject_list', 'subject_list'), 
              ('side', 'side')
             ]
            )
           ]
          )



#PLAN:# resample everything to 176x144x128 and 0.35 mm iso
#Pre-preprocessing
#I am creating a synthetic (template) dataset that we can register/normalise every participant to.
#This will have the average intensity profile and correct resolution/dimensions.
# For completeness, i am doing it like this:
# c3d mprage_to_chunktemp_left.nii.gz -type float -resample 176x144x128 -interpolation sinc -o mprage_left_resample_test.nii.gz
# c3d mprage_left_resample_test.nii.gz -type float -resample-mm 0.35x0.35x0.35mm -interpolation sinc -o mprage_left_resample_test_with_iso_sinc.nii.gz
#then, I average them all
#AverageImages of all of these datasets:
#for side in left right ; do
#AverageImages 3 ${side}_mprage_average_DEEPSEACAT_initial_template.nii.gz 1 *${side}_mprage_left_resample_test_with_iso_sinc.nii.gz ;
#done
#This is the initial template for the template creation (needs to be done for left/right and for MPRAGE and TSE.
#Then create the template using antsMultivariateTemplateConstruction2.sh (which will have the correct resolution and size, needs to be done for left and right)
#antsMultivariateTemplateConstruction2.sh -d 3 -i 3 -k 2 -f 4x2x1 -s 2x1x0vox -q 30x20x4 -t SyN \
# -z left_mprage_average_DEEPSEACAT_initial_template.nii.gz -z right_tse_average_DEEPSEACAT_initial_template.nii.gz \
#  -m MI -c 5 -o right_ right_template_input.csv

#where right_templateInput.csv contains

#subjectA_t1chunk.nii.gz,subjectA_t2chunk.nii.gz
#subjectB_t1chunk.nii.gz,subjectB_t2chunk.nii.gz

#then we are left with a template for left and right TSE and MPRAGE chunks.
#we will need to include these in the atlases. Maybe we can create a new streamlined atlas with only the required files?

###NIPYPELINE STARTS HERE
#once we have these templates, we can use flirt (FSL) to resample our input data to the template images (MPRAGE and TSE chunks)

#step one is to resample the MPRAGE to the template image
#1) the commandline is flirt -in mprage_chunk_right.nii.gz -ref right_template0.nii.gz -applyxfm -usesqform -applyisoxfm 0.35 -interp sinc -datatype float -out out.nii.gz (nipype handles this)
#2) resmpale the TSE to be the same as the mprage using flirt
#flirt -in tse.nii.gz -ref mprage_to_chunktemp_left.nii.gz -applyxfm -usesqform -out tse_chunk_test.nii.gz
#3) do the same for the segmentation.
#normalise all using c3d
#c3d -histmatch to template

#pad (not yet) and not needed.
                                       
    

################
## templates  ##
################
#left right
#set up here    

    
    


##############
## Step One ##
##############

mprage_flirt_n = MapNode(fsl.FLIRT(uses_qform=True, apply_xfm=True, #applyisoxfm, interp sinc
                  name='mprage_flirt_n', iterfield=['in_file'])
wf.connect([(selectfiles, mprage_flirt_n, [('{side}_template_mprage_chunk', 'reference')])])
wf.connect([(selectfiles, mprage_flirt_n, [('{side}_mprage_chunk', 'in_file')])])
#wf.connect([(selectfiles, flirt_n, [('', 'out_file')])])

############
## Step 2 ##
############
tse_flirt_n = MapNode(fsl.FLIRT(uses_qform=True, apply_xfm=True, #applyisoxfm, interp sinc
                  name='tse_flirt_n', iterfield=['in_file'])
wf.connect([(selectfiles, mprage_flirt_n, [('out_file', 'reference')])]) #check that this works FIXME
wf.connect([(selectfiles, tse_flirt_n, [('tse', 'in_file')])])

##############
##  Step 3  ##
##############

segmentation_n = MapNode(fsl.FLIRT(uses_qform=True, apply_xfm=True, #applyisoxfm, interp NEAREST
                  name='segment_flirt_n', iterfield=['in_file'])
wf.connect([(selectfiles, mprage_flirt_n, [('out_file', 'reference')])]) #check that this works FIXME
wf.connect([(selectfiles, segmentation_n, [('seg', 'in_file')])])

##############
##  Step 4  ##
##############

normalise_mprage_n = MapNode(c3d(histmatch=True,
                  name='normalise_mprage_n', iterfield=['in_file'])
wf.connect([(selectfiles, mprage_flirt_n, [('out_file', 'in_file')])]) #check that this works FIXME
#include the reference mprage here
normalise_tse_n = MapNode(c3d(histmatch=True,
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