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
					     -i 1 \
					     -g 0.20 \
					     -k 2 \
					     -t SyN \
					     -n 0 \
					     -z ${workdir}/${side}_averaged_tse_resampled-0.35mmIso.nii.gz \
					     -z ${workdir}/${side}_averaged_mprage_resampled-0.35mmIso.nii.gz \
					     -m CC \
					     -c 5 \
					     -o ${side}_ ${workdir}/${side}_template_input.csv 
done
cd ${workdir}
cp ${workdir}/${side}_template/${side}_template0.nii.gz ${workdir}/${side}_tse_template_resampled-0.35mmIso_padded_176x144x128.nii.gz
cp ${workdir}/${side}_template/${side}_template1.nii.gz ${workdir}/${side}_mprage_template_resampled-0.35mmIso_padded_176x144x128.nii.gz


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
	fslmaths ${workdir}/${side}_${seg}_template_resampled-0.35mmIso_padded_176x144x128_rescaled.nii.gz -sub ${mean} ${side}_${seg}_template_resampled-0.35mmIso_padded_176x144x128_rescaled_0mean.nii.gz
	std=`fslstats ${side}_${seg}_template_resampled-0.35mmIso_padded_176x144x128_rescaled_0mean.nii.gz -s | awk '{print $1}'`
	fslmaths ${workdir}/${side}_${seg}_template_resampled-0.35mmIso_padded_176x144x128_rescaled_0mean.nii.gz -div ${std} ${side}_${seg}_template_resampled-0.35mmIso_padded_176x144x128_rescaled_0meanUv.nii.gz
    done
done
