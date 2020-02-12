#!/usr/bin/env bash

#rm error*
#rm output_*

#BSUB -q normal
#BSUB -W 00:30
#BSUB -n 1
#BSUB -R "span[ptile=1]"
#BSUB -gpu "num=1:j_exclusive=yes"
#BSUB -e "./error.%J.er"
#BSUB -o "./output_%J.out"
#BSUB -J testbench

# Load the required modules
#module load python/3.6.1
#module load python numba cuda numpy scipy loopy openmpi/4.0.1-gcc_5.4.0-cuda_10.0.130
#module load openmpi/4.0.1-gcc_5.4.0-cuda_10.0.130
#module load numba/0.45.1
#ml load numba/0.39.0
#module load llvmlite/0.30.0 
#ml numpy 
#ml scipy/1.2.1
#ml loopy/2017.2
# module load tensorflow/1.12.0-gcc_5.4.0-cuda_10.0.130
# module load scikit-image/0.14.2
# module load opencv/3.4.5-gcc_5.4.0-cuda_10.0.130
# module load opencv-python/3.4.5

# Run the program
#mpirun python ./cudanumba_hackathon.py
#mpirun python ./numba_oscillator.py
#mpirun python ./numba_kuramoto.py
#mpirun python ./numba_kuratmoto_step-5.py
#mpirun nvprof --print-gpu-trace python ./parsweepa.py
#mpirun python ./parsweepa.py
mpirun python ./parsweepam.py
