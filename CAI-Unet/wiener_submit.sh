#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=Build-test
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --gres=gpu:tesla:1
#SBATCH --mem=40000
#SBATCH -o compile_out.txt
#SBATCH -e compile_error.txt
# SBATCH --partition=gpu
#SBATCH --time=0-02:00:00

#source activate /scratch/cai/DEEPSEACAT/
#module load gnu7/7.3.0
#module load anaconda/3.6
#module load openmpi-3.0.0-gcc-4.8.5-6k5odm2
module load cuda/10.0.130

nvidia-smi

srun python Unet-build-test.py
#srun --pty bash -i
