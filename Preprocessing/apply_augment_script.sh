#!/bin/bash
#wrapper around augment script because I'm lazy
#will make 100 random augmented hippo things then apply warps to mprage tse and seg
#then move the whole thing to new dataset 
#dataset location
dataset=/30days/uqtshaw/DEEPSEACAT/data_config_flipped
#have to do it twice because my brain wont work out how to keep the count the same.
count=61
for participant in {000..025} ; do
    cd ${dataset}/${participant}_left
    cp seg_left.nii.gz mprage_left_seg.nii.gz
    cp tse_left.nii.gz mprage_left_tse.nii.gz
    python /30days/uqtshaw/DEEPSEACAT/augment_hippocampus.py mprage_left.nii.gz
    rm mprage_left_tse.nii.gz mprage_left_seg.nii.gz
    #move loop for new segmentations
    for x in {0000..0099} ; do
        mkdir ../0${count}_left
        mv ./generated/g_mprage_left_v${x}.nii.gz ../0${count}_left/mprage_left.nii.gz
        mv ./generated/g_mprage_left_v${x}_labels.nii.gz ../0${count}_left/seg_left.nii.gz
        mv ./generated/g_mprage_left_v${x}_tse.nii.gz ../0${count}_left/tse_left.nii.gz
        let "count++"
    done
    rm -r ./generated
done

count=61
for participant in {000..025} ; do
    cd ${dataset}/${participant}_right
    cp seg_right.nii.gz mprage_right_seg.nii.gz
    cp tse_right.nii.gz mprage_right_tse.nii.gz
    python /30days/uqtshaw/DEEPSEACAT/augment_hippocampus.py mprage_right.nii.gz 
    rm mprage_right_tse.nii.gz mprage_right_seg.nii.gz
    #move loop for new segmentations
    for x in {0000..0099} ; do
        mkdir ../00${count}_right
        mv ./generated/g_mprage_right_v${x}.nii.gz ../00${count}_right/mprage_right.nii.gz
        mv ./generated/g_mprage_right_v${x}_labels.nii.gz ../00${count}_right/seg_right.nii.gz
        mv ./generated/g_mprage_right_v${x}_tse.nii.gz ../00${count}_right/tse_right.nii.gz
        let "count++"
    done
    rm -r generated
done
#now do the mag ones
#count is the number done so far (2600 + 60)
count=2661
for participant in {026..060} ; do
    cd ${dataset}/${participant}_left
    cp mag_seg_left.nii.gz mag_mprage_left_seg.nii.gz
    cp mag_tse_left.nii.gz mag_mprage_left_tse.nii.gz
    python /30days/uqtshaw/DEEPSEACAT/augment_hippocampus.py mag_mprage_left.nii.gz
    rm mag_mprage_left_tse.nii.gz mag_mprage_left_seg.nii.gz
    #move loop for new segmentations
    for x in {0000..0099} ; do
        mkdir ../0${count}_left
        mv ./generated/g_mag_mprage_left_v${x}.nii.gz ../0${count}_left/mag_mprage_left.nii.gz
        mv ./generated/g_mprage_left_v${x}_labels.nii.gz ../0${count}_left/mag_seg_left.nii.gz
        mv ./generated/g_mprage_left_v${x}_tse.nii.gz ../0${count}_left/mag_tse_left.nii.gz
        let "count++"
    done
    rm -r ./generated
done
count=2661
for participant in {026..061} ; do
    cd ${dataset}/${participant}_right
    cp mag_seg_right.nii.gz mag_mprage_right_seg.nii.gz
    cp mag_tse_right.nii.gz mag_mprage_right_tse.nii.gz
    python /30days/uqtshaw/DEEPSEACAT/augment_hippocampus.py mprage_right.nii.gz
    rm mprage_right_tse.nii.gz mprage_right_seg.nii.gz
    #move loop for new segmentations
    for x in {0000..0099} ; do
        mkdir ../0${count}_right
        mv ./generated/g_mprage_right_v${x}.nii.gz ../0${count}_right/mag_mprage_right.nii.gz
        mv ./generated/g_mprage_right_v${x}_labels.nii.gz ../0${count}_right/mag_seg_right.nii.gz
        mv ./generated/g_mprage_right_v${x}_tse.nii.gz ../0${count}_right/mag_tse_right.nii.gz
        let "count++"
    done
    rm -r generated
done