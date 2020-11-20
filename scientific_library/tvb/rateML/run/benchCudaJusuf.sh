#!/usr/bin/env bash

#SBATCH -A slns
#SBATCH --partition=gpus
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o ./output.out
#SBATCH -e ./error.er
#SBATCH --time=04:30:00
#SBATCH -J RateML

# Run the program
srun python ./__main__.py --model mdlrun --bench bencharg -c couplings -s speeds


