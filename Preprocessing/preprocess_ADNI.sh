#!/bin/bash

## Overview of the ADNI preprocessing steps ##
## this is not really accurate but the steps are there
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

#Thomas Shaw 02/12/20
subjName=$1
source ~/.bashrc
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=12
export NSLOTS=12
echo $TMPDIR
raw_data_dir=/30days/${USER}/ADNI_bids/bids_pruned_20200928/
data_dir=/30days/${USER}/ADNI/

for ss in ses-01 ses-02 ses-03; do
    if [[ -d ${raw_data_dir}/${subjName}/${ss} ]]; then
        rsync -rv $raw_data_dir/${subjName}/${ss}/anat $TMPDIR
        mkdir -p $TMPDIR/$subjName
        chmod 744 $TMPDIR -R
        mv $TMPDIR/anat/${subjName}_${ss}_*_run-1_T2w.nii.gz $TMPDIR/$subjName/${subjName}_${ss}_acq-tsehippoTraToLongaxis_run-1_T2w.nii.gz
        mv $TMPDIR/anat/${subjName}_${ss}_acq-*_run-1_T1w.nii.gz $TMPDIR/$subjName/${subjName}_${ss}_acq-mp2rage-UNIDEN_run-1_T1w.nii.gz
        module load singularity/2.5.1
        singularity="singularity exec --bind $TMPDIR:/TMPDIR --pwd /TMPDIR/ $data_dir/ants_fsl_robex_20180524.simg"
        t1w=/TMPDIR/${subjName}/${subjName}_${ss}_acq-mp2rage-UNIDEN_run-1_T1w.nii.gz #this isn't used
        tse1=/TMPDIR/${subjName}/${subjName}_${ss}_acq-tsehippoTraToLongaxis_run-1_T2w.nii.gz
        simg_input_dir=/TMPDIR/${subjName}
        if [[ ! -e $TMPDIR/${subjName}/${subjName}_${ss}*T1w.nii.gz ]]; then
            echo "Missing T1w for ${subjName}_${ss}" >>${data_dir}/preprocessing_error_log.txt
        fi
        if [[ ! -e $TMPDIR/${subjName}/${subjName}_${ss}_acq-tsehippoTraToLongaxis_run-1_T2w.nii.gz ]]; then
            echo "Missing tse1 for ${subjName}_${ss}" >>${data_dir}/preprocessing_error_log.txt
        fi
        #make output dir
        mkdir -p $data_dir/${subjName}
        #initial check to see if files exist, if so, exit loop
        if [[ ! -e ${data_dir}/${subjName}/${subjName}_${ss}_T1w_N4corrected_norm_brain_preproc.nii.gz ]]; then
            
            #initial skull strip
            
            echo "running robex for ${subjName}_${ss} T1w"
            $singularity runROBEX.sh ${simg_input_dir}/${subjName}_${ss}_acq-mp2rage-UNIDEN_run-1_T1w.nii.gz ${simg_input_dir}/${subjName}_${ss}_T1w_brain_preproc.nii.gz ${simg_input_dir}/${subjName}_${ss}_T1w_brainmask.nii.gz
            
            #bias correct t1
            if [[ ! -e ${data_dir}/${subjName}/${subjName}_${ss}_T1w_N4corrected_norm_brain_preproc.nii.gz ]]; then
                
                echo "running N4 for T1w for ${subjName}_${ss} T1w"
                $singularity N4BiasFieldCorrection -d 3 -b [1x1x1,3] -c '[50x50x40x30,0.00000001]' -i ${simg_input_dir}/${subjName}_${ss}_acq-mp2rage-UNIDEN_run-1_T1w.nii.gz -x ${simg_input_dir}/${subjName}_${ss}_T1w_brainmask.nii.gz -r 1 -o ${simg_input_dir}/${subjName}_${ss}_T1w_N4corrected_preproc.nii.gz --verbose 1 -s 2
                
                echo "re-running N4 for T1w for ${subjName}_${ss} T1w after resampling mask to whole image"
                $singularity antsApplyTransforms -d 3 -i ${simg_input_dir}/${subjName}_${ss}_T1w_brainmask.nii.gz -r ${simg_input_dir}/${subjName}_${ss}_acq-mp2rage-UNIDEN_run-1_T1w.nii.gz -n LanczosWindowedSinc -o ${simg_input_dir}/${subjName}_${ss}_T1w_brainmask.nii.gz
                $singularity N4BiasFieldCorrection -d 3 -b [1x1x1,3] -c '[50x50x40x30,0.00000001]' -i ${simg_input_dir}/${subjName}_${ss}_acq-mp2rage-UNIDEN_run-1_T1w.nii.gz -x ${simg_input_dir}/${subjName}_${ss}_T1w_brainmask.nii.gz -r 1 -o ${simg_input_dir}/${subjName}_${ss}_T1w_N4corrected_preproc.nii.gz --verbose 1 -s 2
                
                #rescale t1
                t1bc=${simg_input_dir}/${subjName}_${ss}_T1w_N4corrected_preproc.nii.gz
                
                echo "running robex again with N4'd T1w and norm intensities for T1w for ${subjName}_${ss} T1w"
                $singularity ImageMath 3 ${simg_input_dir}/${subjName}_${ss}_T1w_N4corrected_norm_preproc.nii.gz RescaleImage ${simg_input_dir}/${subjName}_${ss}_T1w_N4corrected_preproc.nii.gz 0 1000
                #skull strip new Bias corrected T1
                $singularity runROBEX.sh ${simg_input_dir}/${subjName}_${ss}_T1w_N4corrected_norm_preproc.nii.gz ${simg_input_dir}/${subjName}_${ss}_T1w_N4corrected_norm_brain_preproc.nii.gz ${simg_input_dir}/${subjName}_${ss}_T1w_brainmask.nii.gz
            fi
            #remove things
            rm $TMPDIR/$subjName/${subjName}_${ss}_T1w_brain_preproc.nii.gz
            rm $TMPDIR/$subjName/${subjName}_${ss}_T1w_N4corrected_preproc.nii.gz
        fi
        #make another loop for nlin bit.
        if [[ ! -e ${data_dir}/${subjName}/${subjName}_${ss}_T2w_NlinMoCo_res-iso.3_N4corrected_denoised_brain_preproc.nii.gz ]]; then
            ######TSE#####
            #apply mask to tse - resample like tse - this is just for BC
            if [[ ! -e ${data_dir}/${subjName}/${subjName}_${ss}_T2w_run-1_brainmask.nii.gz ]]; then
                echo " running apply transforms from T1 to TSE for brainmask for ${subjName}_${ss}"
                $singularity antsApplyTransforms -d 3 -i ${simg_input_dir}/${subjName}_${ss}_T1w_brainmask.nii.gz -r $tse1 -n NearestNeighbor -o ${simg_input_dir}/${subjName}_${ss}_T2w_run-1_brainmask.nii.gz
            fi
            #Bias correction - use mask created -
            for x in "1" ; do
                
                echo "running TSE ${x} N4"
                #N4
                $singularity N4BiasFieldCorrection -d 3 -b [1x1x1,3] -c '[50x50x40x30,0.00000001]' -i ${simg_input_dir}/${subjName}_${ss}_acq-tsehippoTraToLongaxis_run-${x}_T2w.nii.gz -x ${simg_input_dir}/${subjName}_${ss}_T2w_run-${x}_brainmask.nii.gz -r 1 -o ${simg_input_dir}/${subjName}_${ss}_T2w_run-${x}_N4corrected_preproc.nii.gz --verbose 1 -s 2
                if [[ ! -e $TMPDIR/$subjName/${subjName}_${ss}_T2w_run-${x}_N4corrected_preproc.nii.gz ]]; then
                    echo "TSE run ${x} did not bias correct for ${subjName}_${ss}, trying without mask " >>${data_dir}/preprocessing_error_log.txt
                    $singularity N4BiasFieldCorrection -d 3 -b [1x1x1,3] -c '[50x50x40x30,0.00000001]' -i ${simg_input_dir}/${subjName}_${ss}_acq-tsehippoTraToLongaxis_run-${x}_T2w.nii.gz -r 1 -o ${simg_input_dir}/${subjName}_${ss}_T2w_run-${x}_N4corrected_preproc.nii.gz --verbose 1 -s 2
                fi
                #normalise intensities of the BC'd tses
                $singularity ImageMath 3 ${simg_input_dir}/${subjName}_${ss}_T2w_run-${x}_N4corrected_norm_preproc.nii.gz RescaleImage ${simg_input_dir}/${subjName}_${ss}_T2w_run-${x}_N4corrected_preproc.nii.gz 0 1000
                
                #interpolation of TSEs -bring all into the same space while minimising interpolation write steps.
                echo "running interpolation"
                $singularity flirt -v -applyisoxfm 0.5 -interp sinc -sincwidth 8 -in ${simg_input_dir}/${subjName}_${ss}_T2w_run-${x}_N4corrected_norm_preproc.nii.gz -ref ${simg_input_dir}/${subjName}_${ss}_T2w_run-${x}_N4corrected_norm_preproc.nii.gz -out ${simg_input_dir}/${subjName}_${ss}_T2w_run-${x}_res-iso.3_N4corrected_norm_preproc.nii.gz
                
                #create new brainmask and brain images.
                echo "running ants apply transforms to create new brainmask of TSE ${x}"
                $singularity antsApplyTransforms -d 3 -i ${simg_input_dir}/${subjName}_${ss}_T1w_brainmask.nii.gz -r ${simg_input_dir}/${subjName}_${ss}_T2w_run-${x}_res-iso.3_N4corrected_norm_preproc.nii.gz -n NearestNeighbor -o ${simg_input_dir}/${subjName}_${ss}_T2w_run-${x}_brainmask.nii.gz
                rm $TMPDIR/$subjName/${subjName}_${ss}_T2w_run-${x}_N4corrected_norm_preproc.nii.gz
                
                #mask the preprocessed TSE.
                echo "masking the pp'd TSE ${x}"
                $singularity ImageMath 3 ${simg_input_dir}/${subjName}_${ss}_T2w_run-${x}_res-iso.3_N4corrected_norm_brain_preproc.nii.gz m ${simg_input_dir}/${subjName}_${ss}_T2w_run-${x}_res-iso.3_N4corrected_norm_preproc.nii.gz ${simg_input_dir}/${subjName}_${ss}_T2w_run-${x}_brainmask.nii.gz
                if [[ ! -e $TMPDIR/$subjName/${subjName}_${ss}_T2w_run-${x}_res-iso.3_N4corrected_norm_brain_preproc.nii.gz ]]; then
                    echo "${subjName}_${ss} TSE ${x} failed preprocessing" >>${data_dir}/preprocessing_error_log.txt
                fi
                # rm brainmasks and other crap
                rm $TMPDIR/$subjName/${subjName}_${ss}_T2w_run-${x}_brainmask.nii.gz
                rm $TMPDIR/$subjName/${subjName}_${ss}_T2w_run-${x}_N4corrected_norm_preproc.nii.gz
                rm $TMPDIR/$subjName/${subjName}_${ss}_T2w_run-${x}_N4corrected_preproc.nii.gz
            done
            if [[ ! -e $TMPDIR/$subjName/${subjName}_${ss}_T1w_N4corrected_norm_preproc.nii.gz ]]; then
                echo "${subjName}_${ss} T1w failed preprocessing" >>${data_dir}/preprocessing_error_log.txt
            fi
            #copy all the things to the data dir
            rsync -rcv $TMPDIR/${subjName} ${data_dir}/
        fi
        
        #copy the files over
        cp -r /30days/uqtshaw/ADNI_atlas $TMPDIR/$subjName/
        #0: register the T2w to the T1w
        $singularity antsRegistrationSyNQuick.sh -d 3 \
        -f ${simg_input_dir}/${subjName}_${ss}_T1w_N4corrected_norm_preproc.nii.gz \
        -m ${simg_input_dir}/${subjName}_${ss}_T2w_run-1_res-iso.3_N4corrected_norm_preproc.nii.gz \
        -t a \
        -o ${simg_input_dir}/${subjName}_${ss}_T2w_run-1_res-iso.3_N4corrected_norm_preproc_to_T1w_
        #1: Register the T1w to the template (non lin)
        if [[ ! -e $TMPDIR/$subjName/${subjName}_${ss}_T1w_N4corrected_norm_preproc_to_template_Warped.nii.gz ]] ; then
            
            $singularity antsRegistrationSyNQuick.sh -d 3 \
            -f ${simg_input_dir}/ADNI_atlas/correct_spacing_t1w_adni_atlas_ashs_space.nii.gz \
            -m ${simg_input_dir}/${subjName}_${ss}_T1w_N4corrected_norm_preproc.nii.gz \
            -o ${simg_input_dir}/${subjName}_${ss}_T1w_N4corrected_norm_preproc_to_template_
        fi
        #2: apply the chunk mask to the input
        for side in left right ; do
            if [[ ! -e $TMPDIR/$subjName/${subjName}_${ss}_T1w_${side}_chunk_inversed.nii.gz ]] ; then
                $singularity antsApplyTransforms -d 3 \
                -i ${simg_input_dir}/${subjName}_${ss}_T1w_N4corrected_norm_preproc_to_template_Warped.nii.gz \
                -r ${simg_input_dir}/ADNI_atlas/refspace_${side}_0.5mm_112x144x124_bin.nii.gz
                -n LanczosWindowedSinc \
                -o ${simg_input_dir}/${subjName}_${ss}_T1w_N4corrected_norm_preproc_to_template_chunk_${side}.nii.gz
                #T2w
                $singularity antsApplyTransforms -d 3 \
                -i ${simg_input_dir}/${subjName}_${ss}_T2w_run-1_res-iso.3_N4corrected_norm_preproc_to_T1w_Warped.nii.gz \
                -r ${simg_input_dir}/ADNI_atlas/refspace_${side}_0.5mm_112x144x124_bin.nii.gz
                -n LanczosWindowedSinc \
                -o ${simg_input_dir}/${subjName}_${ss}_T2w_run-1_res-iso.3_N4corrected_norm_preproc_to_template_chunk_${side}.nii.gz \
                -t ${simg_input_dir}/${subjName}_${ss}_T1w_N4corrected_norm_preproc_to_template_1Warp.nii.gz \
                -t ${simg_input_dir}/${subjName}_${ss}_T1w_N4corrected_norm_preproc_to_template_0GenericAffine.mat
                
                #9: denoise chunks
                $singularity DenoiseImage -d 3 -n Rician -i ${simg_input_dir}/${subjName}_${ss}_T1w_N4corrected_norm_preproc_to_template_chunk_${side}.nii.gz -o ${simg_input_dir}/${subjName}_${ss}_T1w_N4corrected_norm_preproc_to_template_chunk_${side}.nii.gz -v
                $singularity DenoiseImage -d 3 -n Rician -i ${simg_input_dir}/${subjName}_${ss}_T1w_N4corrected_norm_preproc_to_template_chunk_${side}.nii.gz -o ${simg_input_dir}/${subjName}_${ss}_T1w_N4corrected_norm_preproc_to_template_chunk_${side}.nii.gz -v
            fi
        done
        
        #11: Normalise TSE and MPRAGE chunks respective to the templates - for francesco
        #move back out of TMPDIR... need to delete all the crap (from RDS - the raw files are still included, need to sort this)
        chmod -R 740 $TMPDIR/
        #mkdir /RDS/Q0535/data/$subjName
        rsync -rcv $TMPDIR/${subjName} ${data_dir}/
        echo "done PP for $subjName_${ss}"
        
    fi
done