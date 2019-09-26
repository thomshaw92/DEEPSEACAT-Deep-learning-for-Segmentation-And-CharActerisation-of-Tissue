#!/bin/bash
#preprocessing for the template files for DEEPSEACAT
#this is fairly hacky. Need to sort out relative paths etc
#but it only needs to be done once so maybe not a problem

#Tom Shaw 25/9/19

#go to the directory near all the files

workdir=/data/fastertemp/uqtshaw/DEEPSEACAT_atlas
mkdir ${workdir}
cd ${workdir}

# average all the images in one of the datasets (that'll do, don't need all of them for an initial template)
for seg in mprage tse ; do
    for side in left right ; do
	AverageImages 3 ${workdir}/${side}_averaged_${seg}.nii.gz 1 ../ashs_atlas_umcutrecht_7t_20170810/train/train*/${seg}_to_chunktemp_${side}.nii.gz
 	#fix up the size (this isn't needed but whatever)
	c3d ${workdir}/${side}_averaged_${seg}.nii.gz \
	    -type float \
	    -resample-mm 0.35x0.35x0.35mm \
	    -interpolation sinc \
	    -o ${workdir}/${side}_averaged_${seg}_resampled-0.35mmIso.nii.gz
	#Rescale intensities between -1 and 1
	ImageMath 3 \
		  ${workdir}/${side}_${seg}_template_resampled-0.35mmIso_rescaled.nii.gz \
		  RescaleImage -1 1 \
		  ${workdir}/${side}_${seg}_template_resampled-0.35mmIso.nii.gz
	#Give the templates zero mean and unit variance.
	std=""
	mean=""
	mean=`fslstats ${x} -m | awk '{print $1}'`
	fslmaths ${workdir}/${side}_${seg}_template_resampled-0.35mmIso_rescaled.nii.gz -sub ${mean} ${workdir}/${side}_${seg}_template_resampled-0.35mmIso_rescaled_0mean.nii.gz
	std=`fslstats ${side}_${seg}_template_resampled-0.35mmIso_padded_176x144x128_rescaled_0mean.nii.gz -s | awk '{print $1}'`
	fslmaths ${workdir}/${side}_${seg}_template_resampled-0.35mmIso_rescaled_0mean.nii.gz -div ${std} ${workdir}/${side}_${seg}_template_resampled-0.35mmIso_rescaled_0meanUv.nii.gz
    done
done



##############################################################
# writing the rest of the code in bash to show as an example #
##############################################################

##############

## Step One ##

##############

#make a directory for the final preprocessed data
mkdir ${workdir}/preprocessing_output

#resize the TSE
for side in left right ; do
    for x in {00..32} ; do
	#fix the size of the TSE and seg
	c3d ${atlas_dir}/train0${x}/tse_native_chunk_${side}.nii.gz \
	    -type float \
	    -resample-mm 0.35x0.35x0.35mm \
	    -interpolation sinc \
	    -o ${workdir}/${x}_${side}_tse_chunk_resampled-0.35mmIso.nii.gz
	c3d ${atlas_dir}/train0${x}/tse_native_chunk_${side}_seg.nii.gz \
	    -type float \
	    -resample-mm 0.35x0.35x0.35mm \
	    -interpolation sinc \
	    -o ${workdir}/${x}_${side}_seg_chunk_resampled-0.35mmIso.nii.gz
	
	##############

	## Step Two ##

	##############
	
	#ants the mprage to the tse. I'll just do the Mag dataset for the moment, the affine doesn't exist for the mag dataset (yay) so i'll let ants do all the work
	
	atlas_dir=/data/fastertemp/uqtshaw/ashs_atlas_magdeburg_7t_20180416/train
	mprage=${atlas_dir}/train0${x}/mprage_to_chunktemp_${side}.nii.gz
	tse=${workdir}/${side}_tse_native_chunk_${side}_resampled-0.35mmIso.nii.gz

	antsRegistrationantsRegistration --dimensionality 3 --float 0 \  
        --output [${workdir}/${x}_${side}_mprage_chunk_resampled-0.35mmIso_,${workdir}/${x}_${side}_mprage_chunk_resampled-0.35mmIsoWarped.nii.gz] \  
        --interpolation BSpline \  
        --use-histogram-matching 0 \  
        --transform Rigid[0.15] \  
        --metric MI[${tse},${mprage},1,32,Regular,0.25] \  
        --convergence [1000x500x250x100,1e-6,10] \  
        --shrink-factors 8x4x2x1 \  
        --smoothing-sigmas 3x2x1x0vox \  
        --metric MI[${tse},${mprage},1,32,Regular,0.25] \  
        --convergence [1000x500x250x100,1e-6,10] \  
        --shrink-factors 8x4x2x1 \  
        --smoothing-sigmas 3x2x1x0vox \    
        --transform SyN[0.1,3,0] \  
        --metric MI[${tse},${mprage},1,32,Regular,0.25] \  
        --convergence [100x70x50x20,1e-6,10] \  
        --shrink-factors 8x4x2x1 \  
        --smoothing-sigmas 3x2x1x0vox \  

	##############

	##  Step 3  ##

	##############
	#pad the TSE, MPRAGE, and segmentation 
	for seg in tse mprage seg ; do
	    c3d ${workdir}/${x}_${side}_${seg}_chunk_resampled-0.35mmIso*.nii.gz  \
		-type float \
		-pad-to 176x144x128 0 \
		-interpolation sinc \
		-o ${workdir}/${x}_${side}_${seg}_chunk_resampled-0.35mmIso_pad-176x144x128.nii.gz

	    ##############

	    ##  Step 4  ##

	    ##############
	    #normalise everything  to template
	    
	    c3d ${workdir}/${side}_${seg}_template_resampled-0.35mmIso_rescaled_0meanUv.nii.gz \
		${workdir}/${x}_${side}_${seg}_chunk_resampled-0.35mmIso_pad-176x144x128.nii.gz \
		-histmatch 5 \
		-type float \
		-interpolation sinc \
		-o ${workdir}/${x}_${side}_${seg}_chunk_resampled-0.35mmIso_pad-176x144x128_norm.nii.gz 
	    
	    #normalise everything to the template
	done
    done
done



