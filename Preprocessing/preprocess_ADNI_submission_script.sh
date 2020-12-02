#!/bin/bash

github_dir=~/scripts/DEEPSEACAT-Deep-learning-for-Segmentation-And-CharActerisation-of-Tissue/

for subjName in `cat ${github_dir}/Preprocessing/subjnames.csv` ; do
       qsub -v SUBJNAME=$subjName ${github_dir}/Preprocessing/preprocess_ADNI_pbs_script.pbs
 done