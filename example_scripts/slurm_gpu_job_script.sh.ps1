#!/bin/bash
# Usage: sbatch slurm-gpu-job-script
# Prepared By: Kai Xi,  Feb 2015
#              help@massive.org.au

# NOTE: To activate a SLURM option, remove the whitespace between the '#' and 'SBATCH'

# To give your job a name, replace "MyJob" with an appropriate name
# SBATCH --job-name=MyJob


# To set a project account for credit charging, 
# SBATCH --account=pmosp


# Request CPU resource for a serial job
# SBATCH --ntasks=1
# SBATCH --ntasks-per-node=1
# SBATCH --cpus-per-task=1

# Request for GPU, 
#
# Option 1: Choose any GPU whatever m2070 or K20
# Note in most cases, 'gpu:N' should match '--ntasks=N'
# SBATCH --gres=gpu:1

# Option 2: Choose GPU flavours, "k20m" or "m2070"
# SBATCH --gres=gpu:m2070:1
# Or
# SBATCH --gres=gpu:k20m:1

# Memory usage (MB)
# SBATCH --mem-per-cpu=4000

# Set your minimum acceptable walltime, format: day-hours:minutes:seconds
# SBATCH --time=0-06:00:00


# To receive an email when job completes or fails
# SBATCH --mail-user=<You Email Address>
# SBATCH --mail-type=END
# SBATCH --mail-type=FAIL


# Set the file for output (stdout)
# SBATCH --output=MyJob-%j.out

# Set the file for error log (stderr)
# SBATCH --error=MyJob-%j.err


# Use reserved node to run job when a node reservation is made for you already
# SBATCH --reservation=reservation_name


# Command to run a gpu job
# For example:
module load cuda/7.0
nvidia-smi
deviceQuery