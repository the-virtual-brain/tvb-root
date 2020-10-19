from mako.template import Template

import os
import sys

# not ideal but avoids modifying the vendored LEMS itself
sys.path.append(os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
from lems.model.model import Model

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

def XSD_validate_XML(file_name):
    ''' Use own validation instead of LEMS because of slight difference in definition file'''
    from lxml import etree
    from urllib.request import urlopen

    # Local XSD file location
    # schema_file = urlopen("file:///home/michiel/Documents/Repos/tvb-root/github/tvb-root/scientific_library/tvb/rateML/rML_v0.xsd")

    # Global XSD file location
    schema_file = urlopen(
        "https://raw.githubusercontent.com/DeLaVlag/tvb-root/xsdvalidation/scientific_library/tvb/rateML/rML_v0.xsd")
    xmlschema = etree.XMLSchema(etree.parse(schema_file))
    print("Validating {0} against {1}".format(file_name, schema_file.geturl()))
    xmlschema.assertValid(etree.parse(file_name))
    print("It's valid!")

def load_model(model_filename, folder=None):
    "Load model from filename"

    fp_xml = lems_file(model_filename, folder)

    model = Model()
    model.import_from_file(fp_xml)
    # modelextended = model.resolve()

    XSD_validate_XML(fp_xml)

    return model

def render_model(model_name, template=None, folder=None):

    model = load_model(model_name, folder)
    template = template or default_template()

    modellist = model.component_types['derivatives']

    # coupling functionality
    couplinglist = list()
    for i, cplists in enumerate(model.component_types):
        if 'coupling' in cplists.name:
            couplinglist.append(cplists)

    # collect total number of exposures combinations.
    expolist = list()
    for i, expo in enumerate(modellist.exposures):
        for chc in expo.dimension:
            expolist.append(chc)

    # only check whether noise is there, if so then activate it
    noisepresent=False
    for ct in (model.component_types):
        if ct.name == 'noise' and ct.description == 'on':
            noisepresent=True

    # start templating
    model_str = template.render(
                            modelname=model_name,
                            const=modellist.constants,
                            dynamics=modellist.dynamics,
                            params=modellist.parameters,
                            derparams=modellist.derived_parameters,
                            coupling=couplinglist,
                            noisepresent=noisepresent,
                            exposures=modellist.exposures
                            )

    return model_str

def cuda_templating(model_filename, folder=None):

    modelfile = os.path.join((os.path.dirname(os.path.abspath(__file__))), 'CUDAmodels', model_filename.lower() + '.c')

    # start templating
    model_str = render_model(model_filename, template=default_template(), folder=folder)

    # write template to file
    with open(modelfile, "w") as f:
        f.writelines(model_str)

if __name__ == '__main__':

    # model_filename = 'Oscillator'
    # model_filename = 'Kuramoto'
    # model_filename = 'Rwongwang'
    model_filename = 'Epileptor'
    cuda_templating(model_filename)
