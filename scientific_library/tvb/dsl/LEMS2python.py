# from models import G2DO
from mako.template import Template 

import argparse

import sys
sys.path.insert(0, '../NeuroML/lems/')
from model.model import Model

# model file location
modelname = 'Oscillator'
fp_xml = 'NeuroML/' + modelname.lower() + '.xml'
# modelfile="../models/python/" + modelname + ".py"
modelfile="../simulator/models/" + modelname.lower() + ".py"

model = Model()
model.import_from_file(fp_xml)
modelextended = model.resolve()


def drift_templating():

    # drift dynamics
    modelist = list()
    modelist.append(model.component_types[modelname])

    # start templating
    template = Template(filename='tmpl8_drift.py')
    model_str = template.render(
                            dfunname=modelname,
                            const=modelist[0].constants,
                            dynamics=modelist[0].dynamics,
                            drift=(1e-3, 1e-3)
                            )
    # write template to file
    with open(modelfile, "w") as f:
        f.writelines(model_str)


def coupling_templating():

    # specify type of coupling TODO make dynamic
    modelcoupling = 'Difference'

    # coupling functionality
    couplinglist = list()
    couplinglist.append(model.component_types['coupling_function'])

    template = Template(filename='tmpl8_coupling.py')
    model_str = template.render(
                            couplingname=modelcoupling,
                            couplingconst=couplinglist[0].constants,
                            couplingparams=couplinglist[0].parameters,
                            couplingreqs=couplinglist[0].requirements,
                            couplingfunctions=couplinglist[0].functions
                            )
    # print(model_str)

    modelfile="../models/python/coupling.py"
    with open(modelfile, "w") as f:
        f.writelines(model_str)


def noise_templating():

    # noise collection
    noisetypes = list()
    for i, item in enumerate(model.component_types):
        if item.name == 'noise' or item.extends == 'noise':
            noisetypes.append(item.name)

    noiselist = list()
    noiseconstantspernoise = [[] for i in range(len(noisetypes))]
    for i, type in enumerate(noisetypes):
        for j, const in enumerate(model.component_types[type].constants):
            noiseconstantspernoise[i].append(const)

    # noisetypes = [x.title() for x in noisetypes]

    template = Template(filename='tmpl8_noise.py')
    model_str = template.render(
                            noiseconst=noiseconstantspernoise,
                            # noiseconst=noiselist[0].constants,
                            noisetypes=noisetypes
    )
    # print(model_str)

    modelfile = "../models/python/noise.py"
    with open(modelfile, "w") as f:
        f.writelines(model_str)


drift_templating()
# coupling_templating()
# noise_templating()