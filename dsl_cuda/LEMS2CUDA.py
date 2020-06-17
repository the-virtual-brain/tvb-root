# from models import G2DO
from mako.template import Template

import os
import sys

for p in sys.path:
    print(p)

import dsl
sys.path.append("{}".format(os.path.dirname(dsl.__file__)))

from lems.model.model import Model

# model file location
# model_filename = 'Oscillator'
# model_filename = 'Kuramoto'
# model_filename = 'Rwongwang'
model_filename = 'Epileptor'


def default_lems_folder():
    here = os.path.dirname(os.path.abspath(__file__))
    xmlpath = os.path.join(here, 'XMLmodels')
    return xmlpath

def lems_file(model_name, folder=None):
    folder = folder or default_lems_folder()
    return os.path.join(folder, model_name.lower() + '_CUDA.xml')

def default_template():
    here = os.path.dirname(os.path.abspath(__file__))
    tmp_filename = os.path.join(here, 'tmpl8_CUDA.py')
    template = Template(filename=tmp_filename)
    return template

def load_model(model_filename):
    "Load model from filename"

    fp_xml = lems_file(model_filename)

    model = Model()
    model.import_from_file(fp_xml)
    # modelextended = model.resolve()

    return model

def render_model(model_name, template=None):
    # drift dynamics
    # modelist = list()
    # modelist.append(model.component_types[modelname])

    model = load_model(model_name)
    template = template or default_template()

    modellist = model.component_types[model_name]

    # coupling functionality
    couplinglist = list()
    # couplinglist.append(model.component_types['coupling_function_pop1'])

    for i, cplists in enumerate(model.component_types):
        if 'coupling' in cplists.name:
            couplinglist.append(cplists)

    # collect all signal amplification factors per state variable.
    # signalampl = list()
    # for i, sig in enumerate(modellist.dynamics.derived_variables):
    #     if 'sig' in sig.name:
    #         signalampl.append(sig)

    # collect total number of exposures combinations.
    expolist = list()
    for i, expo in enumerate(modellist.exposures):
        for chc in expo.choices:
            expolist.append(chc)

    # print((couplinglist[0].dynamics.derived_variables['pre'].expression))
    #
    # for m in range(len(couplinglist)):
    #     # print((m))
    #     for k in (couplinglist[m].functions):
    #         print(k)

    # only check whether noise is there, if so then activate it
    noisepresent=False
    for ct in (model.component_types):
        if ct.name == 'noise' and ct.description == 'on':
            noisepresent=True

    # start templating
    # template = Template(filename='tmpl8_CUDA.py')
    model_str = template.render(
                            modelname=model_name,
                            const=modellist.constants,
                            dynamics=modellist.dynamics,
                            params=modellist.parameters,
                            derparams=modellist.derived_parameters,
                            coupling=couplinglist,
                            noisepresent=noisepresent,
                            expolist=expolist
                            )

    return model_str

def cuda_templating(model_filename):

    modelfile = "{}{}{}{}".format(os.path.dirname(dsl.__file__), '/dsl_cuda/CUDAmodels/', model_filename.lower(), '.c')

    # start templating
    model_str = render_model(model_filename, template=default_template())

    # write template to file
    with open(modelfile, "w") as f:
        f.writelines(model_str)


cuda_templating(model_filename)