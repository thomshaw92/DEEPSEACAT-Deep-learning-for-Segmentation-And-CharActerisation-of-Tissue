#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=leaky_ReLu
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --gres=gpu:tesla-smx2:1
#SBATCH --mem=40000
#SBATCH -o compile_out_test_12000_20200810leaky_ReLu.txt
#SBATCH -e compile_error_test_12000_20200810leaky_ReLu.txt
#SBATCH --time=4-4:00:00

source /scratch/cai/tom_shaw/miniconda3/etc/profile.d/conda.sh
module load cuda/10.0.130
conda activate /scratch/cai/tom_shaw/miniconda3/
srun python /scratch/cai/tom_shaw/DEEPSEACAT-Deep-learning-for-Segmentation-And-CharActerisation-of-Tissue/main.py