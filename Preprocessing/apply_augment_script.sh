#!/bin/bash
#wrapper around augment script because I'm lazy
#will make 100 random augmented hippo things then apply warps to mprage tse and seg
#then move the whole thing to new dataset 
#dataset location
dataset=/30days/uqtshaw/DEEPSEACAT/data_vikings
#Do it for every set of participants
#make sure you rename the mag_participants first and then make 

script=/30days/uqtshaw/DEEPSEACAT/DEEPSEACAT-Deep-learning-for-Segmentation-And-CharActerisation-of-Tissue/Preprocessing/augment_hippocampus.py

for side in left right ; do 
    count=0
    #TEST
    for participant in '006' '011' '017' '023' '031' '042' ; do
	cd ${dataset}/${participant}_${side}
	cp seg_${side}.nii.gz mprage_${side}_seg.nii.gz
	cp tse_${side}.nii.gz mprage_${side}_tse.nii.gz
	python ${script} mprage_${side}.nii.gz
	rm mprage_${side}_tse.nii.gz mprage_${side}_seg.nii.gz
	#move loop for new segmentations
	for x in {0000..0099} ; do
            mkdir -p ../../TEST/0${count}_${side}
            mv ./generated/g_mprage_${side}_v${x}.nii.gz ../../TEST/0${count}_${side}/mprage_${side}.nii.gz
            mv ./generated/g_mprage_${side}_v${x}_labels.nii.gz ../../TEST/0${count}_${side}/seg_${side}.nii.gz
            mv ./generated/g_mprage_${side}_v${x}_tse.nii.gz ../../TEST/0${count}_${side}/tse_${side}.nii.gz
            let "count++"
	done
	rm -r ./generated
    done
done
#validate
for side in left right ; do 
    count=0
    for participant in '003' '007' '013' '019' '029' '038'; do
	cd ${dataset}/${participant}_${side}
	cp seg_${side}.nii.gz mprage_${side}_seg.nii.gz
	cp tse_${side}.nii.gz mprage_${side}_tse.nii.gz
	python  ${script} mprage_${side}.nii.gz
	rm mprage_${side}_tse.nii.gz mprage_${side}_seg.nii.gz
	#move loop for new segmentations
	for x in {0000..0099} ; do
            mkdir -p ../../VALIDATE/0${count}_${side}
            mv ./generated/g_mprage_${side}_v${x}.nii.gz ../../VALIDATE/0${count}_${side}/mprage_${side}.nii.gz
            mv ./generated/g_mprage_${side}_v${x}_labels.nii.gz ../../VALIDATE/0${count}_${side}/seg_${side}.nii.gz
            mv ./generated/g_mprage_${side}_v${x}_tse.nii.gz ../../VALIDATE/0${count}_${side}/tse_${side}.nii.gz
            let "count++"
	done
	rm -r ./generated
    done
done
#TRAIN (include if statement that will increment count if the directory doesn't exist in order to keep the count consistent between missing data sets.)
for side in left right ; do 
    count=0
    for participant in '001' '004' '009' '010' '012' '014' '020' '021' '022' '025' '028' '030' '033' '035' '040' '043' '048' '049' '050' '051' '052' '054' '055' '056' '057' '058' '044' '053' '000' '002' '005' '008' '015' '016' '018' '024' '026' '027' '034' '036' '037' '039' '046' '059' '047' ; do
	if [[ -d ${dataset}/${participant}_${side} ]] ; then 
	cd ${dataset}/${participant}_${side}
	cp seg_${side}.nii.gz mprage_${side}_seg.nii.gz
	cp tse_${side}.nii.gz mprage_${side}_tse.nii.gz
	python  ${script} mprage_${side}.nii.gz
	rm mprage_${side}_tse.nii.gz mprage_${side}_seg.nii.gz
	#move loop for new segmentations
	for x in {0000..0099} ; do
            mkdir -p ../../TRAIN/0${count}_${side}
            mv ./generated/g_mprage_${side}_v${x}.nii.gz ../../TRAIN/0${count}_${side}/mprage_${side}.nii.gz
            mv ./generated/g_mprage_${side}_v${x}_labels.nii.gz ../../TRAIN/0${count}_${side}/seg_${side}.nii.gz
            mv ./generated/g_mprage_${side}_v${x}_tse.nii.gz ../../TRAIN/0${count}_${side}/tse_${side}.nii.gz
            let "count++"
	done
	rm -r ./generated
	else
	    for x in {0000..0099} ; do
	       let "count++"	
	    done
	fi	
    done
done

#TRAIN incomplete (left or right only) datasets...

