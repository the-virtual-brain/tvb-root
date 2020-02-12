#!/usr/bin/env bash

#BSUB -q normal
#BSUB -W 00:30
#BSUB -n 1
#BSUB -R "span[ptile=1]"
#BSUB -gpu "num=1:j_exclusive=yes"
##BSUB -e "./error.%J.er"
##BSUB -o "./output_%J.out"
#BSUB -e "./error.er"
#BSUB -o "./output.out"
#BSUB -J testbench

# Run the program
#mpirun python ./cudanumba_hackathon.py
#mpirun python ./numba_oscillator.py
#mpirun python ./numba_kuramoto.py
#mpirun python ./numba_kuratmoto_step-5.py
#mpirun nvprof --print-gpu-trace python ./parsweepa.py
#mpirun python ./parsweepa.py
#mpirun python ./parsweepam.py
#mpirun cuda-memcheck python ./tvbRegCudaNumba.py -b bencharg
mpirun python ./tvbRegCudaNumba.py -b bencharg

