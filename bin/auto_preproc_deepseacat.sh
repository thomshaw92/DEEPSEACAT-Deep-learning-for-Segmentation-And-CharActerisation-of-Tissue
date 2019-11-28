#!/bin/bash
#quick and nasty script for the automatic data preprocessing - this can be redone in nipype later
#includes QSM28, 7TEA, and TOMCAT, and ADNI to come

cd /data/fastertemp/uqtshaw/7Tea/derivatives/ashs_xs
mkdir DEEPSEACAT_concat
for subjName in sub-* ; do
    mkdir DEEPSEACAT_concat/$subjName
    #do the registration
    c3d ${subjName}/tse.nii.gz -resample 100x100x100% -region 20x20x0% 60x60x100% -type float -o ${subjName}/tse_iso.nii.gz
    greedy -d 3 -a -dof 6 -m MI -n 100x100x10 -i  ${subjName}/tse_iso.nii.gz  ${subjName}/mprage.nii.gz -ia-identity -o ${subjName}/tse_to_mprage_greedy.mat
    c3d ${subjName}/tse_iso.nii.gz ${subjName}/mprage.nii.gz -reslice-matrix ${subjName}/tse_to_mprage_greedy.mat -interpolation Sinc -o ${subjName}/mprage_to_tse_greedy.nii.gz
    #then reslice the mprage and seg to the tse chunks
    for side in left right ; do
	c3d ${subjName}/tse_native_chunk_${side}.nii.gz ${subjName}/mprage_to_tse_greedy.nii.gz -reslice-identity -interpolation Sinc -o DEEPSEACAT_concat/${subjName}/mprage_native_chunk_${side}.nii.gz
	antsApplyTransforms -d 3 -i ${subjName}/final/${subjName}_${side}_lfseg_corr_usegray.nii.gz -r ${subjName}/tse_native_chunk_${side}.nii.gz -o DEEPSEACAT_concat/${subjName}/seg_${side}.nii.gz -n MultiLabel
	cp ${subjName}/tse_native_chunk_${side}.nii.gz DEEPSEACAT_concat/${subjName}/
    done
done
tar -cvzf DEEPSEACAT_concat.tar.gz DEEPSEACAT_concat
cp -r DEEPSEACAT_concat.tar.gz /winmounts/uqtshaw/uq-research/DEEPSEACAT-Q1219/data/automatic_data/qsm28/

cd /data/fasttemp/uqtshaw/tomcat/data/derivatives/2_xs_ashs
mkdir DEEPSEACAT_concat
for subjName in sub-* ; do
    mkdir DEEPSEACAT_concat/$subjName
    #do the registration
    c3d ${subjName}/tse.nii.gz -resample 100x100x100% -region 20x20x0% 60x60x100% -type float -o ${subjName}/tse_iso.nii.gz
    greedy -d 3 -a -dof 6 -m MI -n 100x100x10 -i  ${subjName}/tse_iso.nii.gz  ${subjName}/mprage.nii.gz -ia-identity -o ${subjName}/tse_to_mprage_greedy.mat
    c3d ${subjName}/tse_iso.nii.gz ${subjName}/mprage.nii.gz -reslice-matrix ${subjName}/tse_to_mprage_greedy.mat -interpolation Sinc -o ${subjName}/mprage_to_tse_greedy.nii.gz
    #then reslice the mprage and seg to the tse chunks
    for side in left right ; do
	c3d ${subjName}/tse_native_chunk_${side}.nii.gz ${subjName}/mprage_to_tse_greedy.nii.gz -reslice-identity -interpolation Sinc -o DEEPSEACAT_concat/${subjName}/mprage_native_chunk_${side}.nii.gz
	antsApplyTransforms -d 3 -i ${subjName}/final/${subjName}_${side}_lfseg_corr_usegray.nii.gz -r ${subjName}/tse_native_chunk_${side}.nii.gz -o DEEPSEACAT_concat/${subjName}/seg_${side}.nii.gz -n MultiLabel
	cp ${subjName}/tse_native_chunk_${side}.nii.gz DEEPSEACAT_concat/${subjName}/
    done
done
tar -cvzf DEEPSEACAT_concat.tar.gz DEEPSEACAT_concat
cp -r DEEPSEACAT_concat.tar.gz /winmounts/uqtshaw/uq-research/DEEPSEACAT-Q1219/data/automatic_data/tomcat/


cd /data/fasttemp/uqtshaw/ashs_native_qsm28
mkdir DEEPSEACAT_concat
for subjName in sub-* ; do
    mkdir DEEPSEACAT_concat/$subjName
    #do the registration
    c3d ${subjName}/${subjName}/1_nlin/tse.nii.gz -resample 100x100x100% -region 20x20x0% 60x60x100% -type float -o ${subjName}/${subjName}/1_nlin/tse_iso.nii.gz
    greedy -d 3 -a -dof 6 -m MI -n 100x100x10 -i ${subjName}/${subjName}/1_nlin/tse_iso.nii.gz ${subjName}/${subjName}/1_nlin/mprage.nii.gz -ia-identity -o ${subjName}/${subjName}/1_nlin/tse_to_mprage_greedy.mat
    c3d ${subjName}/${subjName}/1_nlin/tse_iso.nii.gz ${subjName}/${subjName}/1_nlin/mprage.nii.gz -reslice-matrix ${subjName}/${subjName}/1_nlin/tse_to_mprage_greedy.mat -interpolation Sinc -o ${subjName}/${subjName}/1_nlin/mprage_to_tse_greedy.nii.gz
    
     #then reslice the mprage and seg to the tse chunks
    for side in left right ; do
	c3d ${subjName}/${subjName}/1_nlin/tse_native_chunk_${side}.nii.gz ${subjName}/${subjName}/1_nlin/mprage_to_tse_greedy.nii.gz -reslice-identity  -interpolation Sinc -o DEEPSEACAT_concat/${subjName}/mprage_native_chunk_${side}.nii.gz
	antsApplyTransforms -d 3 -i ${subjName}/${subjName}/1_nlin/final/${subjName}_${side}_lfseg_corr_usegray.nii.gz -r ${subjName}/${subjName}/1_nlin/tse_native_chunk_${side}.nii.gz -o DEEPSEACAT_concat/${subjName}/seg_${side}.nii.gz -n MultiLabel
	cp ${subjName}/${subjName}/1_nlin/tse_native_chunk_${side}.nii.gz DEEPSEACAT_concat/${subjName}/
    done
done
tar -cvzf DEEPSEACAT_concat.tar.gz DEEPSEACAT_concat
cp -r DEEPSEACAT_concat.tar.gz /winmounts/uqtshaw/uq-research/DEEPSEACAT-Q1219/data/automatic_data/qsm28/


cd /data/fastertemp/uqtshaw/optimex_ashs
mkdir DEEPSEACAT_concat
for subjName in sub-* ; do
    mkdir DEEPSEACAT_concat/$subjName
    #do the registration
    c3d ${subjName}/tse.nii.gz -resample 100x100x100% -region 20x20x0% 60x60x100% -type float -o ${subjName}/tse_iso.nii.gz
    greedy -d 3 -a -dof 6 -m MI -n 100x100x10 -i  ${subjName}/tse_iso.nii.gz  ${subjName}/mprage.nii.gz -ia-identity -o ${subjName}/tse_to_mprage_greedy.mat
    c3d ${subjName}/tse_iso.nii.gz ${subjName}/mprage.nii.gz -reslice-matrix ${subjName}/tse_to_mprage_greedy.mat -interpolation Sinc -o ${subjName}/mprage_to_tse_greedy.nii.gz
    #then reslice the mprage and seg to the tse chunks
    for side in left right ; do
	c3d ${subjName}/tse_native_chunk_${side}.nii.gz ${subjName}/mprage_to_tse_greedy.nii.gz -reslice-identity -interpolation Sinc -o DEEPSEACAT_concat/${subjName}/mprage_native_chunk_${side}.nii.gz
	antsApplyTransforms -d 3 -i ${subjName}/final/${subjName}_${side}_lfseg_corr_usegray.nii.gz -r ${subjName}/tse_native_chunk_${side}.nii.gz -o DEEPSEACAT_concat/${subjName}/seg_${side}.nii.gz -n MultiLabel
	cp ${subjName}/tse_native_chunk_${side}.nii.gz DEEPSEACAT_concat/${subjName}/
    done
done
mkdir /winmounts/uqtshaw/uq-research/DEEPSEACAT-Q1219/data/automatic_data/optimex
tar -cvzf DEEPSEACAT_concat.tar.gz DEEPSEACAT_concat
cp -r DEEPSEACAT_concat.tar.gz /winmounts/uqtshaw/uq-research/DEEPSEACAT-Q1219/data/automatic_data/optimex/
