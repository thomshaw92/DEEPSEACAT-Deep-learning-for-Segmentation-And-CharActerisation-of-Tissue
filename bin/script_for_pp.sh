#!/bin/bash
#preprocessing for the template files for DEEPSEACAT
#this is fairly hacky. Need to sort out relative paths etc
#but it only needs to be done once so maybe not a problem

#Tom Shaw 25/9/19

#go to the directory near all the files
mag_atlasdir=/data/fastertemp/uqtshaw/ashs_atlas_magdeburg_7t_20180416/
umc_atlasdir=/data/fastertemp/uqtshaw/ashs_atlas_umcutrecht_7t_20170810/
workdir=/data/fastertemp/uqtshaw/DEEPSEACAT_atlas

mkdir ${workdir}
cd ${workdir}
export ITK_GLOBAL_DEFAUL_NUMBER_OF_THREADS=35
######################################################################
#This all needs to be done only once so commented out for the moment##
######################################################################

#first add the files to a CSV

#for side in left right ; do
#    for x in {00..32} ; do
#	echo "/data/fastertemp/uqtshaw/ashs_atlas_magdeburg_7t_20180416/train/train0${x}/tse_to_chunktemp_${side}.nii.gz,/data/fastertemp/uqtshaw/ashs_atlas_magdeburg_7t_20180416/train/train0${x}/mprage_to_chunktemp_${side}.nii.gz">>${side}_template_input.csv
#    done
#    for x in  {00..25} ; do
#	echo "/data/fastertemp/uqtshaw/ashs_atlas_umcutrecht_7t_20170810/train/train0${x}/tse_to_chunktemp_${side}.nii.gz,/data/fastertemp/uqtshaw/ashs_atlas_umcutrecht_7t_20170810/train/train0${x}/mprage_to_chunktemp_${side}.nii.gz">>${side}_template_input.csv
#    done
#done

#make a template of the average morphology. Prolly only need one iteration as they are already all registered. 
#for side in left right ; do
#    mkdir ${workdir}/${side}_template
#   cd ${workdir}/${side}_template
#   antsMultivariateTemplateConstruction2.sh -d 3 \
#					     -i 4 \
#					     -g 0.20 \
#					     -k 2 \
#					     -t SyN \
#					     -n 0 \
#					     -c 5 \
#					     -r 1 \
#					     -y 0 \
#					     -o ${side}_ ${workdir}/${side}_template_input.csv 

#    cp ${workdir}/${side}_template/${side}_template0.nii.gz ${workdir}/${side}_tse_template.nii.gz
#    cp ${workdir}/${side}_template/${side}_template1.nii.gz ${workdir}/${side}_mprage_template.nii.gz
#done

#cd ${workdir}

#for side in "left" "right" ; do
#    for seg in "tse" "mprage" ; do     	
	#resize
#	c3d ${workdir}/${side}_${seg}_template.nii.gz \
#	    -type float \
#	    -resample-mm 0.35x0.35x0.35mm \
#	    -interpolation sinc \
#	    -o ${workdir}/${side}_${seg}_template_resampled-0.35mmIso.nii.gz

	#Rescale intensities between -1 and 1
#	ImageMath \
#	    3 \
#	    ${workdir}/${side}_${seg}_template_resampled-0.35mmIso_rescaled.nii.gz \
#	    RescaleImage ${workdir}/${side}_${seg}_template_resampled-0.35mmIso.nii.gz -1 1
	#Give the templates zero mean and unit variance.
#	std=""
#	mean=""
#	mean=`fslstats ${workdir}/${side}_${seg}_template_resampled-0.35mmIso_rescaled.nii.gz -m | awk '{print $1}'`
#	fslmaths ${workdir}/${side}_${seg}_template_resampled-0.35mmIso_rescaled.nii.gz -sub ${mean} ${workdir}/${side}_${seg}_template_resampled-0.35mmIso_rescaled_0mean.nii.gz
#	std=`fslstats ${workdir}/${side}_${seg}_template_resampled-0.35mmIso_rescaled_0mean.nii.gz -s | awk '{print $1}'`
#	fslmaths ${workdir}/${side}_${seg}_template_resampled-0.35mmIso_rescaled_0mean.nii.gz -div ${std} ${workdir}/${side}_${seg}_template_resampled-0.35mmIso_rescaled_0meanUv.nii.gz
	#Pad the templates to correct size, create a mask.
#	c3d ${workdir}/${side}_${seg}_template_resampled-0.35mmIso_rescaled_0meanUv.nii.gz \
#	    -type float \
#	    -interpolation sinc \
#	    -pad-to 176x144x128 0 \
#	    -o ${workdir}/${side}_${seg}_template_resampled-0.35mmIso_rescaled_0meanUv_pad-176x144x128.nii.gz
#	fslmaths ${workdir}/${side}_${seg}_template_resampled-0.35mmIso_rescaled_0meanUv_pad-176x144x128.nii.gz \
#	    -add 1 \
#	    -bin  \
#	    ${workdir}/${side}_${seg}_template_resampled-0.35mmIso_rescaled_0meanUv_pad-176x144x128-bin.nii.gz
#   done
#done

#Subject Processing#

##############

## Step One ##

##############

#make a directory for the final preprocessed data
mkdir ${workdir}/preprocessing_output

#Step one, seeing as the UMC dataset is already close to isotropic, we will use it as our standard.
#Native chunks for TSE contain the segmentation, so we will keep them in this space.
#We need to buff it up to 0.35mm iso anyway to get the resolution consistent across all data

for x in  {00..25} ; do
    c3d ${umc_atlasdir}/train/train0${x}/tse.nii.gz \
	-type float \
	-interpolation sinc \
	-resample-mm 0.35x0.35x0.35mm \
	-o ${umc_atlasdir}/train/train0${x}/tse_resampled-0.35mmIso.nii.gz

    #But the chunks are different sizes, so we will resize them to the correct size.
    #pad the tse_native_chunk to the correct size, binarize, 
    for side in left right ; do 
	c3d ${umc_atlasdir}/train/train0${x}/tse_native_chunk_${side}.nii.gz \
	    -type float \
	    -interpolation sinc \
	    -resample-mm 0.35x0.35x0.35mm \
	    -pad-to 176x144x128 0 \
	    -binarize \
	    -o ${umc_atlasdir}/train/train0${x}/tse_native_chunk_${side}_pad-176x144x128_bin.nii.gz


	#then multiply by the original TSE to get the same sized chunks across the dataset.
	#reslice first
	c3d ${umc_atlasdir}/train/train0${x}/tse.nii.gz ${umc_atlasdir}/train/train0${x}/tse_native_chunk_${side}_pad-176x144x128_bin.nii.gz \
	     -reslice-identity \
	     -type float \
	     -interpolation sinc \
	     -o ${umc_atlasdir}/train/train0${x}/tse_native_chunk_${side}_pad-176x144x128_bin_reslice_to_whole.nii.gz 
	
	c3d ${umc_atlasdir}/train/train0${x}/tse.nii.gz ${umc_atlasdir}/train/train0${x}/tse_native_chunk_${side}_pad-176x144x128_bin_reslice_to_whole.nii.gz \
	    -multiply \
	    -type float \
	    -interpolation sinc \
	    -o ${umc_atlasdir}/train/train0${x}/tse_native_chunk_${side}_nopad-176x144x128.nii.gz
	
    done
done

#for the madeburg one, the data is anisotropic, so we need to resample the scans into the space of the umc data.

#making the TSE template in 1mm iso because otherwise it is too big.
########################################################################
##############This only needs to be done once so i am commenting it####
########################################################################
#c3d ${umc_atlasdir}/train/train_${x}/tse.nii.gz \
#    -type float \
#    -resample-mm 1x1x1mm \
#    -interpolation sinc \
#    -o ${umc_atlasdir}/train/train_${x}/tse_resampled_1mmIso.nii.gz

#Then create a TSE template because I don't want to bias my registrations to one subject. Also only needs to be done once
#tse template should be released with package

#antsMultivariateTemplateConstruction2.sh  \
#    -d 3 \
#    -i 4 \
#    -k 1 \
#    -g 0.25 \
#    -t SyN \
#    -n 1 \
#    -r 1 \
#    -c 5 \
#    -o ${umc_atlasdir}/template/tse_template_ \
#    ${umc_atlasdir}/train/train0*/tse_resampled_1mmIso.nii.gz

#Then resample the template to 0.35mm iso (again, only done once.)


#c3d ${umc_atlasdir}/template/tse_template_template0.nii.gz \
#    -type float \
#    -resample-mm 0.35x0.35x0.35mm \
#    -interpolation sinc \
#    -o ${umc_atlasdir}/template/tse_template_resampled_0.35mm.nii.gz

#then rigidly register the TSE.nii scans of every magdeburg participant to the template of the UMC peeps.

for side in left right ; do
    for x in {00..32} ; do
	tse=${mag_atlasdir}/train/train0${x}/tse.nii.gz
	template=${umc_atlasdir}/template/tse_template_resampled_0.35mm.nii.gz
	antsRegistrationSyNQuick.sh \
	    -d 3 \
	    -t r \
	    -f ${template} \
	    -m ${tse} \
	    -n 30 \
	    -o ${mag_atlasdir}/train/train0${x}/tse_to_umc_space_rigid_
 

	#Then we need to do the same as before, cut the TSE chunk out of the TSE

	#antsApplyTransforms to the affine just created to the segmentation and the native chunk
	antsApplyTransforms -d 3 \
			    -i ${mag_atlasdir}/train/train0${x}/tse_native_chunk_${side}.nii.gz \
			    -r ${mag_atlasdir}/train/train0${x}/tse_to_umc_space_rigid_Warped.nii.gz \
			    -o ${mag_atlasdir}/train/train0${x}/tse_native_chunk_${side}_warped_to_umc.nii.gz \
			    -t ${mag_atlasdir}/train/train0${x}/tse_to_umc_space_rigid_*eneric*ffine* \
			    -n BSpline
		    
	antsApplyTransforms -d 3 \
			    -i ${mag_atlasdir}/train/train0${x}/tse_native_chunk_${side}_seg.nii.gz \
			    -r ${mag_atlasdir}/train/train0${x}/tse_to_umc_space_rigid_Warped.nii.gz \
			    -o ${mag_atlasdir}/train/train0${x}/tse_native_chunk_${side}_seg_warped_to_umc.nii.gz \
			    -t ${mag_atlasdir}/train/train0${x}/tse_to_umc_space_rigid_*eneric*ffine* \
			    -n NearestNeighbor
	
	c3d ${mag_atlasdir}/train/train0${x}/tse_native_chunk_${side}_warped_to_umc.nii.gz \
	    -type float \
	    -interpolation sinc \
	    -resample-mm 0.35x0.35x0.35mm \
	    -pad-to 176x144x128 0 \
	    -binarize \
	    -o ${mag_atlasdir}/train/train0${x}/tse_native_chunk_${side}_pad-176x144x128_bin.nii.gz
	
	#then multiply by the original TSE to get the same sized chunks across the dataset.
	#reslice first
	c3d ${mag_atlasdir}/train/train0${x}/tse_native_chunk_${side}_warped_to_umc.nii.gz ${mag_atlasdir}/train/train0${x}/tse_native_chunk_${side}_pad-176x144x128_bin.nii.gz \
	     -reslice-identity \
	     -type float \
	     -interpolation sinc \
	     -o ${mag_atlasdir}/train/train0${x}/tse_native_chunk_${side}_pad-176x144x128_bin_reslice_to_whole.nii.gz 
	
	c3d ${mag_atlasdir}/train/train0${x}/tse_native_chunk_${side}_warped_to_umc.nii.gz ${mag_atlasdir}/train/train0${x}/tse_native_chunk_${side}_pad-176x144x128_bin_reslice_to_whole.nii.gz \
	    -multiply \
	    -type float \
	    -interpolation sinc \
	    -o ${mag_atlasdir}/train/train0${x}/tse_native_chunk_${side}_nopad-176x144x128.nii.gz
	#now everything is in the same space and size except the segmentations and mprage	

    done
done
for side in left right ; do
    for x in {00..32} ; do
	
	#fix the size of the seg to be the same as the nopad files.
	#the mag one is in the same space as the tse native but needs to have the same size
	c3d ${mag_atlasdir}/train/train0${x}/tse_native_chunk_${side}_nopad-176x144x128.nii.gz ${mag_atlasdir}/train/train0${x}/tse_native_chunk_${side}_seg_warped_to_umc.nii.gz \
	    -reslice-identity \
	    -type float \
	    -interpolation  NearestNeighbor \
	    -o ${mag_atlasdir}/train0${x}/${x}_${side}_seg_chunk_resampled-0.35mmIso_nopad-176x144x128.nii.gz
    done
done

for side in left right ; do
    for x in {00..25} ; do
	c3d  ${umc_atlasdir}/train/train0${x}/tse_native_chunk_${side}_nopad-176x144x128.nii.gz ${umc_atlasdir}/train0${x}/tse_native_chunk_${side}_seg.nii.gz \
	     -type float \
	     -interpolation NearestNeighbor \
	     -o ${umc_atlasdir}/train0${x}/${x}_${side}_seg_chunk_resampled-0.35mmIso_nopad-176x144x128.nii.gz
    done
done

##############

## Step Two ##

##############

#ants the mprage to the tse chunk (the original 0.35mm one that is not padded).
# This is now different for the mag and the UMC datasets 

for side in left right ; do
    for x in {00..32} ; do 
	#mag
	mprage=${mag_atlasdir}/train/train0${x}/mprage_to_chunktemp_${side}.nii.gz
	tse=${mag_atlasdir}/train/train0${x}/tse_native_chunk_${side}_warped_to_umc.nii.gz
	antsRegistrationSyNQuick.sh \
	    -d 3 \
	    -f ${tse} \
	    -m ${mprage} \
	    -n 30 \
	    -o ${mag_atlasdir}/train/train0${x}/mprage_to_tse_

    done
done

for side in left right ; do
    for x in {00..25} ; do 
	#umc
	mprage=${umc_atlasdir}/train/train0${x}/mprage_to_chunktemp_${side}.nii.gz
	tse= ${umc_atlasdir}/train/train0${x}/tse_resampled-0.35mmIso.nii.gz
	antsRegistrationSyNQuick.sh \
	    -d 3 \
	    -f ${tse} \
	    -m ${mprage} \
	    -n 30 \
	    -o ${umc_atlasdir}/train/train0${x}/mprage_to_tse_


    done
done

##############

##  Step 3  ##

##############
#normalise everything  to templates
#umc
for x in {00..25} ; do  
    for side in left right ; do    
	#tse
	c3d ${workdir}/${side}_tse_template_resampled-0.35mmIso_rescaled_0meanUv.nii.gz ${umc_atlasdir}/train/train0${x}/tse_native_chunk_${side}_nopad-176x144x128.nii.gz \
	    -histmatch 5 \
	    -type float \
	    -interpolation sinc \
	    -o ${workdir}/preprocessing_output/umc_${x}_${side}_tse_chunk_resampled-0.35mmIso_nopad-176x144x128_norm0meanUv.nii.gz
	#mprage (include padding after histmatch)
	c3d ${workdir}/${side}_mprage_template_resampled-0.35mmIso_rescaled_0meanUv.nii.gz ${umc_atlasdir}/train/train0${x}/mprage_to_tse_*arped.nii.gz \
	    -histmatch 5 \
	    -type float \
	    -interpolation sinc \
	    -pad-to 176x144x128 0 \
	    -o ${workdir}/preprocessing_output/umc_${x}_${side}_mprage_chunk_resampled-0.35mmIso_pad-176x144x128_norm0meanUv.nii.gz

	#cp the seg to the preprocessing_output folder
 cp ${umc_atlasdir}/train0${x}/${x}_${side}_seg_chunk_resampled-0.35mmIso_nopad-176x144x128.nii.gz ${workdir}/preprocessing_output/umc_${x}_${side}_seg_chunk_resampled-0.35mmIso_nopad-176x144x128.nii.gz
	
    done
done
#mag
for x in {00..32} ; do  
    for side in left right ; do    
	#tse
	c3d ${workdir}/${side}_tse_template_resampled-0.35mmIso_rescaled_0meanUv.nii.gz ${mag_atlasdir}/train/train0${x}/tse_native_chunk_${side}_nopad-176x144x128.nii.gz \
	    -histmatch 5 \
	    -type float \
	    -interpolation sinc \
	    -o ${workdir}/preprocessing_output/mag_${x}_${side}_tse_chunk_resampled-0.35mmIso_nopad-176x144x128_norm0meanUv.nii.gz

	#mprage (include padding after histmatch)
	c3d ${workdir}/${side}_mprage_template_resampled-0.35mmIso_rescaled_0meanUv.nii.gz ${mag_atlasdir}/train/train0${x}/mprage_to_tse_*arped.nii.gz \
	    -histmatch 5 \
	    -type float \
	    -interpolation sinc \
	    -pad-to 176x144x128 0 \
	    -o ${workdir}/preprocessing_output/mag_${x}_${side}_mprage_chunk_resampled-0.35mmIso_pad-176x144x128_norm0meanUv.nii.gz

	#cp the seg to the preprocessing_output folder
	cp ${mag_atlasdir}/train0${x}/${x}_${side}_seg_chunk_resampled-0.35mmIso_nopad-176x144x128.nii.gz ${workdir}/preprocessing_output/mag_${x}_${side}_seg_chunk_resampled-0.35mmIso_nopad-176x144x128.nii.gz	
    done
done

#full reg code if that isn't in nipype:
	#antsRegistration --dimensionality 3 --float 0 \  
        #--output [${mag_atlasdir}/train/train${x}/tse_to_umc_space_,${mag_atlasdir}/train/train${x}/tse_to_umc_space_Warped.nii.gz] \  
        #--interpolation BSpline \  
        #--use-histogram-matching 0 \  
        #--transform Rigid[0.15] \  
        #--metric MI[${template},${tse},1,32,Regular,0.25] \  
        #--convergence [1000x500x250x100,1e-6,10] \  
        #--shrink-factors 8x4x2x1 \  
        #--smoothing-sigmas 3x2x1x0vox \  
        #--metric MI[${template},${tse},1,32,Regular,0.25] \  
        #--convergence [1000x500x250x100,1e-6,10] \  
        #--shrink-factors 8x4x2x1 \  
        #--smoothing-sigmas 3x2x1x0vox \    
        #--transform SyN[0.1,3,0] \  
        #--metric CC[${template},${tse},1,4] \  
        #--convergence [100x70x50x20,1e-6,10] \  
        #--shrink-factors 8x4x2x1 \  
        #--smoothing-sigmas 3x2x1x0vox \
