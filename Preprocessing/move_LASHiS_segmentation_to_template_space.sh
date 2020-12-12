#!/bin/bash
#simple script - registers image to template version of image (t1w)
#apply transforms to the LASHiS segmentation with -r as the chunk TSE
#???
#2020/12/11
#I wrote this because I deleted the warp files to save space.... stupid.

ml ants
input_dir=/30days/uqtshaw/ADNI_data_deepseacat/LASHiS/
output_dir=/30days/uqtshaw/ADNI_data_deepseacat/ADNI_data/
reference_dir=/30days/uqtshaw/ADNI_data_deepseacat/preprocessing

for subjName in `cat ${input_dir}/files.csv ` ; do
	cd $input_dir
	if [[ ! -d ${subjName} ]] ; then tar xvzf ${subjName}.tar.gz ; fi
	#if [[ ! -e ${reference_dir}/${subjName}/${subjName}_ses-01_lashis_to_template_1Warp.nii.gz ]] ; then
	      antsRegistrationSyNQuick.sh -d 3 -f ${reference_dir}/correct_spacing_t1w_adni_atlas_ashs_space.nii.gz -m ${input_dir}/${subjName}/${subjName}_ses-01_T1w_N4corrected_norm_preproc_denoised_0/${subjName}_ses-01_T1w_N4corrected_norm_preproc_denoised/mprage.nii.gz -n 50 -o ${reference_dir}/${subjName}/${subjName}_ses-01_lashis_to_template_
	#fi
	for side in left right ; do
	    antsApplyTransforms -d 3 -i ${input_dir}/${subjName}/${subjName}_ses-01_T1w_N4corrected_norm_preproc_denoised_0/${subjName}_ses-01_T1w_N4corrected_norm_preproc_denoised/final/${subjName}_ses-01_T1w_N4corrected_norm_preproc_denoised_${side}_lfseg_corr_usegray.nii.gz -r ${output_dir}/${subjName}/${subjName}_ses-01_T2w_run-1_res-iso.5_N4corrected_norm_preproc_to_template_chunk_${side}.nii.gz -n MultiLabel -o ${output_dir}/${subjName}/${subjName}_ses-01_T2w_run-1_res-iso.5_N4corrected_norm_preproc_to_template_chunk_${side}_segmentation.nii.gz -t ${reference_dir}/${subjName}/${subjName}_ses-01_lashis_to_template_1Warp.nii.gz -t ${reference_dir}/${subjName}/${subjName}_ses-01_lashis_to_template_0GenericAffine.mat 
	done	
		antsRegistrationSyNQuick.sh -d 3 -f ${reference_dir}/correct_spacing_t1w_adni_atlas_ashs_space.nii.gz -m ${input_dir}/${subjName}/${subjName}_ses-02_T1w_N4corrected_norm_preproc_denoised_1/${subjName}_ses-02_T1w_N4corrected_norm_preproc_denoised/mprage.nii.gz -n 50 -o ${reference_dir}/${subjName}/${subjName}_ses-02_lashis_to_template_
	for side in left right ; do
	    antsApplyTransforms -d 3 -i ${input_dir}/${subjName}/${subjName}_ses-02_T1w_N4corrected_norm_preproc_denoised_1/${subjName}_ses-02_T1w_N4corrected_norm_preproc_denoised/final/${subjName}_ses-02_T1w_N4corrected_norm_preproc_denoised_${side}_lfseg_corr_usegray.nii.gz -r ${output_dir}/${subjName}/${subjName}_ses-02_T2w_run-1_res-iso.5_N4corrected_norm_preproc_to_template_chunk_${side}.nii.gz -n MultiLabel -o ${output_dir}/${subjName}/${subjName}_ses-02_T2w_run-1_res-iso.5_N4corrected_norm_preproc_to_template_chunk_${side}_segmentation.nii.gz -t ${reference_dir}/${subjName}/${subjName}_ses-02_lashis_to_template_1Warp.nii.gz -t ${reference_dir}/${subjName}/${subjName}_ses-02_lashis_to_template_0GenericAffine.mat
	done
		antsRegistrationSyNQuick.sh -d 3 -f ${reference_dir}/correct_spacing_t1w_adni_atlas_ashs_space.nii.gz -m ${input_dir}/${subjName}/${subjName}_ses-03_T1w_N4corrected_norm_preproc_denoised_2/${subjName}_ses-03_T1w_N4corrected_norm_preproc_denoised/mprage.nii.gz -n 50 -o ${reference_dir}/${subjName}/${subjName}_ses-03_lashis_to_template_
	for side in left right ; do
	    antsApplyTransforms -d 3 -i ${input_dir}/${subjName}/${subjName}_ses-03_T1w_N4corrected_norm_preproc_denoised_2/${subjName}_ses-03_T1w_N4corrected_norm_preproc_denoised/final/${subjName}_ses-03_T1w_N4corrected_norm_preproc_denoised_${side}_lfseg_corr_usegray.nii.gz -r ${output_dir}/${subjName}/${subjName}_ses-03_T2w_run-1_res-iso.5_N4corrected_norm_preproc_to_template_chunk_${side}.nii.gz -n MultiLabel -o ${output_dir}/${subjName}/${subjName}_ses-03_T2w_run-1_res-iso.5_N4corrected_norm_preproc_to_template_chunk_${side}_segmentation.nii.gz -t ${reference_dir}/${subjName}/${subjName}_ses-03_lashis_to_template_1Warp.nii.gz -t ${reference_dir}/${subjName}/${subjName}_ses-03_lashis_to_template_0GenericAffine.mat
	done
	rsync -rcv  ${output_dir}/${subjName} /winmounts/uqtshaw/data.cai.uq.edu.au/DEEPSEACAT-Q1219/data/ADNI_data/
done
