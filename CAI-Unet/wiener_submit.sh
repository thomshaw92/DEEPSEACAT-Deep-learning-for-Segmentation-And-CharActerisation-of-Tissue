#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=Build-test
#SBATCH -n 1
# SBATCH -c 1
#SBATCH --gres=gpu:tesla-smx2:1
#SBATCH --mem=40000
#SBATCH -o compile_out_v2.txt
#SBATCH -e compile_error_v2.txt
# SBATCH --partition=gpu
#SBATCH --time=0-12:00:00

#source activate /scratch/cai/DEEPSEACAT/
#module load gnu7/7.3.0
#module load anaconda/3.6
#module load openmpi-3.0.0-gcc-4.8.5-6k5odm2
module load cuda/10.0.130
conda activate /scratch/cai/DEEPSEACAT/

#nvidia-smi
#cp -r afm01/Q1/Q1219/data/data_for_network_rearranged/ /scratch/cai/DEEPSEACAT/data/new_data_config/ 

srun python Unet-build-test.py

#cp /scratch/cai/DEEPSEACAT/data/new_data_config/ /afm01/Q1/Q1219/data/data_for_network_rearranged/
#rm /scratch/cai/DEEPSEACAT/data/new_data_config/

#srun --pty bash -i
