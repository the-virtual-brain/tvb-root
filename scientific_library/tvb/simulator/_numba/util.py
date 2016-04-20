
import os

# http://numba.pydata.org/numba-doc/dev/cuda/simulator.html
try:
    CUDA_SIM = int(os.environ['NUMBA_ENABLE_CUDASIM']) == 1
except:
    CUDA_SIM = False