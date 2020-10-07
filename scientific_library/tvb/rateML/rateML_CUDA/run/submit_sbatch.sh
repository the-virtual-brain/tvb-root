#!/usr/bin/env bash

#SBATCH --partition=dp-esb
#SBATCH -A type1_1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o output.out
#SBATCH -e error.er
#SBATCH --time=00:30:00
#SBATCH -J rateML

# Run the program
srun python ./ --bench bencharg


