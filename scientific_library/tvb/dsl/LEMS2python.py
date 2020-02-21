# from models import G2DO
from mako.template import Template 

import sys
# sys.path.insert(0, 'NeuroML/lems/')
from model.model import Model

def drift_templating(target):

    def montbrio():
        modelname = 'Theta2D'
        filename = 'montbrio'
        return modelname, filename

    def epileptor():
        modelname = 'Epileptor'
        filename = 'epileptor'
        return modelname, filename

    def oscillator():
        modelname = 'Generic2dOscillator' # is also the class name
        filename = 'oscillator' # TVB output file name
        print('here')
        return modelname, filename

    def wong_wang():
        modelname = 'ReducedWongWang' # is also the class name
        filename = 'wong_wang' # TVB output file name
        return modelname, filename

    def kuramoto():
        modelname = 'Kuramoto'  # is also the class name
        filename = 'kuramoto'  # TVB output file name
        return modelname, filename

    switcher = {
        'Kuramoto': kuramoto,
        'ReducedWongWang': wong_wang,
        'Generic2dOscillator': oscillator,
        'Epileptor': epileptor,
        'Montbrio': montbrio
    }

    func = switcher.get(target, 'invalid model choice')
    modelname, filename = func()
    print(modelname)

    fp_xml = 'NeuroML/' + filename.lower() + '.xml'
    # modelfile="../models/python/" + modelname + ".py"
    # place results directly into tvb model directory
    modelfile = "../simulator/models/" + filename.lower() + ".py"

    model = Model()
    model.import_from_file(fp_xml)
    modelextended = model.resolve()

    # drift dynamics
    modelist = list()
    modelist.append(model.component_types[modelname])
    # print((modelist[0].dynamics.conditional_derived_variables['ctmp0'].cases[1]))

    # start templating
    template = Template(filename='tmpl8_drift.py')
    model_str = template.render(
                            dfunname=modelname,
                            const=modelist[0].constants,
                            dynamics=modelist[0].dynamics,
                            exposures=modelist[0].exposures
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

if __name__ == '__main__':
    drift_templating('Montbrio')

# coupling_templating()
# noise_templating()

