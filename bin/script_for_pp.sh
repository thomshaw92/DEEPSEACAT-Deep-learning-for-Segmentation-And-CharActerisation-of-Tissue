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

#first add the files to a CSV

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
					     -i 4 \
					     -g 0.20 \
					     -k 2 \
					     -t SyN \
					     -n 0 \
					     -c 5 \
					     -r 1 \
					     -y 0 \
					     -o ${side}_ ${workdir}/${side}_template_input.csv 
    
    cp ${workdir}/${side}_template/${side}_template0.nii.gz ${workdir}/${side}_tse_template.nii.gz
    cp ${workdir}/${side}_template/${side}_template1.nii.gz ${workdir}/${side}_mprage_template.nii.gz
done

cd ${workdir}

for side in  left right ; do 
    #resize
    c3d ${workdir}/${side}_${seg}_template.nii.gz \
	-type float \
	-resample-mm 0.35x0.35x0.35mm \
	-interpolation sinc \
	-o ${workdir}/${side}_${seg}_template_resampled-0.35mmIso.nii.gz

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
    std=`fslstats ${side}_${seg}_template_resampled-0.35mmIso_rescaled_0mean.nii.gz -s | awk '{print $1}'`
    fslmaths ${workdir}/${side}_${seg}_template_resampled-0.35mmIso_rescaled_0mean.nii.gz -div ${std} ${workdir}/${side}_${seg}_template_resampled-0.35mmIso_rescaled_0meanUv.nii.gz

    #Pad the templates to correct size, create a mask.
    c3d ${workdir}/${side}_${seg}_template_resampled-0.35mmIso_rescaled_0meanUv.nii.gz \
	-type float \
	-interpolation sinc \
	-pad-to 176x144x128 0 \
	-o ${workdir}/${side}_${seg}_template_resampled-0.35mmIso_rescaled_0meanUv_pad-176x144x128.nii.gz

    c3d ${workdir}/${side}_${seg}_template_resampled-0.35mmIso_rescaled_0meanUv_pad-176x144x128.nii.gz \
	-type float \
	-add 1 \
	-interpolation sinc \
	-binarize  \
	-o ${workdir}/${side}_${seg}_template_resampled-0.35mmIso_rescaled_0meanUv_pad-176x144x128-bin.nii.gz
done


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
    c3d ${umc_atlasdir}/train/train_${x}/tse.nii.gz \
	-type float \
	-interpolation sinc \
	-resample-mm 0.35x0.35x0.35mm \
	-o ${umc_atlasdir}/train/train_${x}/tse_resampled-0.35mmIso.nii.gz

    #But the chunks are different sizes, so we will resize them to the correct size.
    #pad the tse_native_chunk to the correct size, binarize, 
    for side in left right ; do 
	c3d ${umc_atlasdir}/train/train_${x}/tse_native_chunk_${side}.nii.gz \
	    -type float \
	    -interpolation sinc \
	    -resample-mm 0.35x0.35x0.35mm \
	    -pad-to 176x144x128 0 \
	    -binarize \
	    -o ${umc_atlasdir}/train/train_${x}/tse_native_chunk_${side}_pad-176x144x128_bin.nii.gz


	#then multiply by the original TSE to get the same sized chunks across the dataset.
	#reslice first
	c3d  ${umc_atlasdir}/train/train_${x}/tse_native_chunk_${side}_pad-176x144x128_bin.nii.gz \
	    ${umc_atlasdir}/train/train_${x}/tse.nii.gz	\
	    -reslice-identity \
	    -type float \
	    -interpolation sinc \
	    ${umc_atlasdir}/train/train_${x}/tse_native_chunk_${side}_pad-176x144x128_bin_reslice_to_whole.nii.gz 
	
	c3d ${umc_atlasdir}/train/train_${x}/tse_native_chunk_${side}_pad-176x144x128_bin_reslice_to_whole.nii.gz \
	    ${umc_atlasdir}/train/train_${x}/tse.nii.gz	\
	    -multiply \
	    -type float \
	    -interpolation sinc \
	    ${umc_atlasdir}/train/train_${x}/tse_native_chunk_${side}_nopad-176x144x128.nii.gz
	
    done
done

    #for the madeburg one, the data is anisotropic, so we need to resample the scans into the space of the umc data.

    #making the TSE template in 1mm iso because otherwise it is too big. This only needs to be done once so i am commenting it
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
	tse=${mag_atlasdir}/train/train${x}/tse.nii.gz
	template=${umc_atlasdir}/template/tse_template_resampled_0.35mm.nii.gz
	antsRegistrationSyNQuick.sh \
	    -d 3 \
	    -t r \
	    -f ${template} \
	    -m ${tse} \
	    -n 30 \
	    -o ${mag_atlasdir}/train/train${x}/tse_to_umc_space_rigid_
    done
done
<<EOF



	#Then we need to do the same as before, cut the TSE chunk out of the TSE

	#antsApplyTransforms to the affine just created to the segmentation and the native chunk
	#code here
	
	# TSE is ${mag_atlasdir}/train/train${x}/tse_to_umc_space_

	#Then chunk the TSE in the new space by again multiplying the resized and binarised mask (just warped) to the warped TSE
	#code here (same as before)

	#now everything is in the same space and size except the segmentations and mprage
	c3d ${mag_atlasdir}/train/train_${x}/tse.nii.gz \
	-type float \
	-interpolation sinc \
	-resample-mm 0.35x0.35x0.35mm \
	-o ${umc_atlasdir}/train/train_${x}/tse_resampled-0.35mmIso.nii.gz

    #But the chunks are different sizes, so we will resize them to the correct size.
    #pad the tse_native_chunk to the correct size, binarize, 
    for side in left right ; do 
	c3d ${umc_atlasdir}/train/train_${x}/tse_native_chunk_${side}.nii.gz \
	    -type float \
	    -interpolation sinc \
	    -resample-mm 0.35x0.35x0.35mm \
	    -pad-to 176x144x128 0 \
	    -binarize \
	    -o ${umc_atlasdir}/train/train_${x}/tse_native_chunk_${side}_pad-176x144x128_bin.nii.gz


	#then multiply by the original TSE to get the same sized chunks across the dataset.
	#reslice first
	c3d  ${umc_atlasdir}/train/train_${x}/tse_native_chunk_${side}_pad-176x144x128_bin.nii.gz \
	    ${umc_atlasdir}/train/train_${x}/tse.nii.gz	\
	    -reslice-identity \
	    -type float \
	    -interpolation sinc \
	    ${umc_atlasdir}/train/train_${x}/tse_native_chunk_${side}_pad-176x144x128_bin_reslice_to_whole.nii.gz 
	
	c3d ${umc_atlasdir}/train/train_${x}/tse_native_chunk_${side}_pad-176x144x128_bin_reslice_to_whole.nii.gz \
	    ${umc_atlasdir}/train/train_${x}/tse.nii.gz	\
	    -multiply \
	    -type float \
	    -interpolation sinc \
	    ${umc_atlasdir}/train/train_${x}/tse_native_chunk_${side}_nopad-176x144x128.nii.gz
	
    done
	
	#fix the size of the seg
	c3d ${mag_atlasdir}/train0${x}/ #THIS FILE NAME IS NOW OUTPUT FROM WARP STEP tse_native_chunk_${side}_seg.nii.gz \
	-type float \
	      -resample-mm 0.35x0.35x0.35mm \
	      -pad-to 176x144x128 0\
	      -interpolation sinc \
	      -o ${mag_atlasdir}/train0${x}/${x}_${side}_seg_chunk_resampled-0.35mmIso_pad-176x144x128.nii.gz
	c3d ${umc_atlasdir}/train0${x}/tse_native_chunk_${side}_seg.nii.gz \
	    -type float \
	    -resample-mm 0.35x0.35x0.35mm \
	    -pad-to 176x144x128 0\
	    -interpolation sinc \
	    -o ${umc_atlasdir}/train0${x}/${x}_${side}_seg_chunk_resampled-0.35mmIso_pad-176x144x128.nii.gz
	
	
	##############

	    ## Step Two ##

	    ##############

	    #ants the mprage to the tse chunk (the original 0.35mm one that is not padded).
	    # This is now different for the mag and the UMC datasets so I'll just do the UMC dataset for the moment,
	    #the affine doesn't exist for the mag dataset (yay) so i'll let ants do all the work


	    mprage=${umc_atlasdir}/train0${x}/mprage_to_chunktemp_${side}.nii.gz
	    ####tse=${}/${side}_tse_native_chunk_${side}_resampled-0.35mmIso.nii.gz

	  

	    ##############

	    ##  Step 3  ##

	    ##############
	    #pad the MPRAGE to make it the same as the TSE (will leave 0s around the edge but that shouldnt matter???????) slkjfdnbsakhbjfvdhblskdnlnv 
	    c3d ${workdir}/${x}_${side}_mprage_chunk_resampled-0.35mmIso*.nii.gz  \
		-type float \
		-pad-to 176x144x128 0 \
		-interpolation sinc \
		-o ${workdir}/${x}_${side}_${seg}_chunk_resampled-0.35mmIso_pad-176x144x128.nii.gz

	    ##############

	    ##  Step 4  ##

	    ##############
	    #normalise everything  to template
	    for seg in mprage tse ; do 
		c3d ${workdir}/${side}_${seg}_template_resampled-0.35mmIso_rescaled_0meanUv.nii.gz \
		    ${workdir}/${x}_${side}_${seg}_chunk_resampled-0.35mmIso_pad-176x144x128.nii.gz \
		    -histmatch 5 \
		    -type float \
		    -interpolation sinc \
		    -o ${workdir}/preprocessing_output/${x}_${side}_${seg}_chunk_resampled-0.35mmIso_pad-176x144x128_norm.nii.gz 
	    done
	done
    done
	

#full code if that isn't in nipype:
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
EOF
