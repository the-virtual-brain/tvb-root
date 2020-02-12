# from models import G2DO
from mako.template import Template

import argparse

import sys
sys.path.insert(0, 'NeuroML/lems/')
# print(sys.path)
from model.model import Model

# model file location
# modelname = 'Oscillator_2D'
modelname = 'Kuramoto'
# modelname = 'rWongWang'
# modelname = 'Epileptor_2D'

fp_xml = 'NeuroML/' + modelname.lower() + '_CUDA' + '.xml'
modelfile="models/CUDA/" + modelname.lower() + ".c"

model = Model()
model.import_from_file(fp_xml)
# modelextended = model.resolve()

def templating():

    # drift dynamics
    # modelist = list()
    # modelist.append(model.component_types[modelname])

    modellist = model.component_types[modelname]

    # coupling functionality
    couplinglist = list()
    # couplinglist.append(model.component_types['coupling_function_pop1'])

    for i, cplists in enumerate(model.component_types):
        if 'coupling' in cplists.name:
            couplinglist.append(cplists)

    # collect all signal amplification factors per state variable.
    signalampl = list()
    for i, sig in enumerate(modellist.dynamics.derived_variables):
        if 'sig' in sig.name:
            signalampl.append(sig)

    # print((couplinglist[1].functions['pre'].value))
    #
    # for m in range(len(couplinglist)):
    #     # print((m))
    #     for k in (couplinglist[m].functions):
    #         print(k)

    # only check whether noise is there, if so then activate it
    # TODO: make more dynamical and add noises of TVB
    for ct in (model.component_types):
        if ct.name == 'noise':
            noisepresent=True


    # start templating
    template = Template(filename='tmpl8_CUDA.py')
    model_str = template.render(
                            modelname=modelname,
                            # const=modelist[0].constants,
                            const=modellist.constants,
                            dynamics=modellist.dynamics,
                            params=modellist.parameters,
                            couplingfunctions=couplinglist[0].functions,
                            glob_couplingBehaviour=couplinglist[0].derived_parameters,
                            coupling=couplinglist,
                            noisepresent=noisepresent,
                            exposures=modellist.exposures,
                            signalampl=signalampl
                            )
    # write template to file
    with open(modelfile, "w") as f:
        f.writelines(model_str)


templating()