#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=20191107_base_Wconfig3
#SBATCH -n 1
# SBATCH -c 1
#SBATCH --gres=gpu:tesla-smx2:1
#SBATCH --mem=40000
#SBATCH -o compile_out_20191107_base_Wconfig3.txt
#SBATCH -e compile_error_20191107_base_Wconfig3.txt
# SBATCH --partition=gpu
#SBATCH --time=4-12:00:00
#SBATCH --mail-user=<mtg15@student.aau.dk>
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


module load cuda/10.0.130
conda activate /scratch/cai/DEEPSEACAT/


srun python Unet-build-test.py

