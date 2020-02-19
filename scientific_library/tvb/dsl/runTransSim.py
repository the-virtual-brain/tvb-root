
import sys, os
# sys.path.append('/home/michiel/Documents/TVB/tvb-root/scientific_library/')
# print(sys.path)
import LEMS2python as templating
import TVB_testsuite.tvbRegCudaNumba as TemplSim
import time

# # options for target:
# Kuramoto
# ReducedWongWang
# Generic2dOscillator
# Epileptor

target="Epileptor"
# make a model template
templating.drift_templating(target)

#     # run tvb with model template
testTemplSim = TemplSim.TVB_test()
testTemplSim.startsim(target)


# target = ["Epileptor", "Kuramoto"]
# for i, trgt in enumerate(target):
#     # make a model template
#     templating.drift_templating(trgt)
#     time.sleep(2)
#     if (i==0):
#         import TVB_testsuite.tvbRegCudaNumba as TemplSim
#     # run tvb with model template
#     testTemplSim = TemplSim.TVB_test()
#     testTemplSim.startsim(trgt)
