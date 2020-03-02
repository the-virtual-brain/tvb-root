
import sys, os
# sys.path.append('/home/michiel/Documents/TVB/tvb-root/scientific_library/')
# print(sys.path)
import LEMS2python as templating
import time

# # options for target:
# Kuramoto
# ReducedWongWang
# Generic2dOscillator
# Epileptor
# Montbrio

# target="ReducedWongWang"
# # make a model template
# templating.regTVB_templating(target)
# # run tvb with model template
# testTemplSim = TemplSim.TVB_test()
# testTemplSim.startsim(target)


# model_target = ["Montbrio", "Epileptor", "Kuramoto", "ReducedWongWang", "Generic2dOscillator"]
model_target = ["MontbrioT", "EpileptorT", "KuramotoT", "ReducedWongWangT", "Generic2dOscillatorT"]
for i, trgt in enumerate(model_target):
    # def montbrio():
    #     modelname = 'Theta2D'
    #     filename = 'montbrio'
    #     return modelname, filename
    #
    # def epileptor():
    #     modelname = 'Epileptor'
    #     filename = 'epileptor'
    #     return modelname, filename
    #
    # def oscillator():
    #     modelname = 'Generic2dOscillator' # is also the class name
    #     filename = 'oscillator' # TVB output file name
    #     return modelname, filename
    #
    # def wong_wang():
    #     modelname = 'ReducedWongWang' # is also the class name
    #     filename = 'wong_wang' # TVB output file name
    #     return modelname, filename
    #
    # def kuramoto():
    #     modelname = 'Kuramoto'  # is also the class name
    #     filename = 'kuramoto'  # TVB output file name
    #     return modelname, filename
    #
    # switcher = {
    #     'Kuramoto': kuramoto,
    #     'ReducedWongWang': wong_wang,
    #     'Generic2dOscillator': oscillator,
    #     'Epileptor': epileptor,
    #     'Montbrio': montbrio
    # }
    #
    # func = switcher.get(trgt, 'invalid model choice')
    # modelname, filename = func()
    print('\n Building and running model:', trgt)
    # make a model template
    # modelname = trgt+'T'
    templating.regTVB_templating(trgt)

    import TVB_testsuite.tvbRegCudaNumba as TemplSim
    # # run tvb with model template
    testTemplSim = TemplSim.TVB_test()
    testTemplSim.startsim(trgt)
