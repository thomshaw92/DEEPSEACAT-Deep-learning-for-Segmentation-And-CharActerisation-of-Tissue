#!/bin/bash
#preprocessing for the template files for DEEPSEACAT
#this is fairly hacky. Need to sort out relative paths etc
#but it only needs to be done once so maybe not a problem

#Tom Shaw 25/9/19

#go to the directory near all the files

workdir=/data/fastertemp/uqtshaw/DEEPSEACAT_atlas
mkdir ${workdir}
cd ${workdir}


#then average all the images in one of the datasets (that'll do, don't need all of them for an initial template)
for seg in mprage tse ; do
    for side in left right ; do
	AverageImages 3 ${workdir}/${side}_averaged_${seg}.nii.gz 1 ../ashs_atlas_umcutrecht_7t_20170810/train/train*/${seg}_to_chunktemp_${side}.nii.gz

	#fix up the size and pad

	c3d ${workdir}/${side}_averaged_${seg}.nii.gz \
	    -type float \
	    -resample-mm 0.35x0.35x0.35mm \
	    -pad-to 176x144x128 0 \
	    -interpolation sinc \
	    -o ${workdir}/${side}_averaged_${seg}_resampled-0.35mmIso.nii.gz

    done
done

#Then create the template 
#first the files to a CSV 
for side in left right ; do
    for x in {00..32} ; do
	echo "/data/fastertemp/uqtshaw/ashs_atlas_magdeburg_7t_20180416/train/train0${x}/tse_to_chunktemp_${side}.nii.gz,/data/fastertemp/uqtshaw/ashs_atlas_magdeburg_7t_20180416/train/train0${x}/mprage_to_chunktemp_${side}.nii.gz">>${side}_template_input.csv
    done
    for x in  {00..25} ; do
	echo "/data/fastertemp/uqtshaw/ashs_atlas_umcutrecht_7t_20170810/train/train0${x}/tse_to_chunktemp_${side}.nii.gz,/data/fastertemp/uqtshaw/ashs_atlas_umcutrecht_7t_20170810/train/train0${x}/mprage_to_chunktemp_${side}.nii.gz">>${side}_template_input.csv
    done
done

#make a template of the average morphology. Prolly only need one iteration as they are already all registered. 
for side in left right ; do
    mkdir ${workdir}/${side}_template
    cd ${workdir}/${side}_template
    antsMultivariateTemplateConstruction2.sh -d 3 \
					     -i 3 \
					     -g 0.20 \
					     -k 2 \
					     -t SyN \
					     -n 0 \
					     -z ${workdir}/${side}_averaged_tse_resampled-0.35mmIso.nii.gz \
					     -z ${workdir}/${side}_averaged_mprage_resampled-0.35mmIso.nii.gz \
					     -m CC \
					     -c 5 \
					     -y 1 \
					     -o ${side}_ ${workdir}/${side}_template_input.csv 
	cd ${workdir}
	cp ${workdir}/${side}_template/${side}_template0.nii.gz ${workdir}/${side}_tse_template_resampled-0.35mmIso_padded_176x144x128.nii.gz
	cp ${workdir}/${side}_template/${side}_template1.nii.gz ${workdir}/${side}_mprage_template_resampled-0.35mmIso_padded_176x144x128.nii.gz
done

#Rescale intensities between -1 and 1
for seg in mprage tse ; do
    for side in left right ; do
	ImageMath 3 \
		   ${workdir}/${side}_${seg}_template_resampled-0.35mmIso_padded_176x144x128_rescaled.nii.gz \
		  RescaleImage -1 1 \
		   ${workdir}/${side}_${seg}_template_resampled-0.35mmIso_padded_176x144x128.nii.gz
    done
done

#Give the templates zero mean and unit variance.
for seg in mprage tse ; do
    for side in left right ; do
	std=""
	mean=""
	mean=`fslstats ${x} -m | awk '{print $1}'`
	fslmaths ${workdir}/${side}_${seg}_template_resampled-0.35mmIso_padded_176x144x128_rescaled.nii.gz -sub ${mean} ${workdir}/${side}_${seg}_template_resampled-0.35mmIso_padded_176x144x128_rescaled_0mean.nii.gz
	std=`fslstats ${side}_${seg}_template_resampled-0.35mmIso_padded_176x144x128_rescaled_0mean.nii.gz -s | awk '{print $1}'`
	fslmaths ${workdir}/${side}_${seg}_template_resampled-0.35mmIso_padded_176x144x128_rescaled_0mean.nii.gz -div ${std} ${workdir}/${side}_${seg}_template_resampled-0.35mmIso_padded_176x144x128_rescaled_0meanUv.nii.gz
    done
done

#../ashs_atlas_umcutrecht_7t_20170810/train/train000/seg_right.nii.gz
#antsRegistrationSyNQuick.sh -d 3 -t a -o tse_test_ants_to_chunk -f ./right_averaged_tse_resampled-0.35mmIso.nii.gz -m ../ashs_atlas_umcutrecht_7t_20170810/train/train000/tse.nii.gz
#antsApplyTransforms -d 3 -o ${movingMaskWarped} -i ${movingMask} -r ${fixedT1} -n NearestNeighbor -t tse_test_ants_to_chunk1Warp.nii.gz -t ${outputPrefix}0GenericAffine.mat


#ok, the mprage was in a completely different space to the segmentation, so we need to warp the mp2rage back to normal space
#this code works to trasform the TSE to the space of the MPRAGE but i can't invert it for some reason. (because the inverse warp isn't included, yay.)
#########
antsApplyTransforms -d 3 -o ../ashs_atlas_umcutrecht_7t_20170810/train/train010/tse_native_chunk_right_to_mprage.nii.gz -i ../ashs_atlas_umcutrecht_7t_20170810/train/train010/tse_native_chunk_right.nii.gz -r ../ashs_atlas_umcutrecht_7t_20170810/train/train010/mprage_to_chunktemp_right.nii.gz -n BSpline -t ../ashs_atlas_umcutrecht_7t_20170810/train/train010/greedy_t1_to_template_right_warp.nii.gz  -t ../ashs_atlas_umcutrecht_7t_20170810/train/train010/ants_t1_to_tempAffine.txt
###########
#OK, step one. Get the MPRAGE to be in the same space as the TSE. We can initialise it with the inverse .mat and do a quickSyn to get it near perfectly aligned with the TSE.
#there is probably no other solution unless we can invert thedeformation field..
#nope that doesn't exist

##############################################################
# writing the rest of the code in bash to show as an example #
##############################################################

##############

## Step One ##

##############

#make a directory for the final preprocessed data
mkdir ${workdir}/preprocessing_inputs

#flirt the mprage to the template using sqform. I'll just do the Mag dataset
atlas_dir=/data/fastertemp/uqtshaw/ashs_atlas_magdeburg_7t_20180416/train

for side in left right ; do
    for x in {00..32} ; do
	flirt -in ${atlas_dir}/train0${x}/mprage_to_chunktemp_${side}.nii.gz \
	      -applyxfm \
	      -usesqform \
	      -interp sinc \
	      -ref ${workdir}/${side}_mprage_template_resampled-0.35mmIso_padded_176x144x128_rescaled_0meanUv.nii.gz \
	      -out ${workdir}/preprocessing_inputs/mag_${x}_mprage_resampled-0.35mmIso_padded_176x144x128.nii.gz
	#Step Two
	#do the TSE
	flirt -in ${atlas_dir}/train0${x}/tse.nii.gz \
	      -applyxfm \
	      -usesqform \
	      -interp sinc \
	      -ref ${workdir}/${side}_tse_template_resampled-0.35mmIso_padded_176x144x128_rescaled_0meanUv.nii.gz \
	      -out ${workdir}/preprocessing_inputs/mag_${x}_tse_resampled-0.35mmIso_padded_176x144x128.nii.gz
	#Step Three
	#Do the Segmentation
	flirt -in ${atlas_dir}/train0${x}/tse.nii.gz \
	      -applyxfm \
	      -usesqform \
	      -interp nearestneighbour \
	      -ref ${workdir}/${side}_tse_template_resampled-0.35mmIso_padded_176x144x128_rescaled_0meanUv.nii.gz \
	      -out ${workdir}/preprocessing_inputs/mag_${x}_seg_resampled-0.35mmIso_padded_176x144x128.nii.gz
    done
done


##############

##  Step 4  ##

##############
#normalise everything to the template

c3d 

normalise_mprage_n = MapNode(c3d(histmatch=True,

                  name='normalise_mprage_n', iterfield=['in_file'])

wf.connect([(selectfiles, mprage_flirt_n, [('out_file', 'in_file')])]) #check that this works FIXME

#include the reference mprage here

normalise_tse_n = MapNode(c3d(histmatch=True,

                  name='normalise_tse_n', iterfield=['in_file'])

wf.connect([(selectfiles, tse_flirt_n, [('out_file', 'in_file')])])

wf.connect #need to include the reference file here




