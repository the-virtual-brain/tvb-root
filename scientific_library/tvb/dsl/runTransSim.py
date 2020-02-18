
import sys, os
# sys.path.append('/home/michiel/Documents/TVB/tvb-root/scientific_library/')
# print(sys.path)

# # options for target:
# Kuramoto
# ReducedWongWang
# Generic2dOscillator
# Epileptor

target = 'ReducedWongWang'
import LEMS2python as templating

templating.drift_templating(target)

import TVB_testsuite.tvbRegCudaNumba as TemplSim

testTemplSim = TemplSim.TVB_test()
testTemplSim.startsim(target)
