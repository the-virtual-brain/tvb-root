# -*- coding: utf-8 -*-

"""
LEMS2python module implements a DSL code generation using a TVB-specific LEMS-based DSL.

.. moduleauthor:: Michiel. A. van der Vlag <m.van.der.vlag@fz-juelich.de>   
.. moduleauthor:: Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>

"""

from mako.template import Template
import tvb
import sys
import os
from .NeuroML.lems.model.model import Model


def default_lems_folder():
    here = os.path.dirname(os.path.abspath(__file__))
    xmlpath = os.path.join(here, 'NeuroML', 'XMLmodels')
    return xmlpath


def lems_file(model_name, folder=None):
    folder = folder or default_lems_folder()
    return os.path.join(folder, model_name.lower() + '.xml')


def load_model(model_filename):
    "Load model from filename"

    fp_xml = lems_file(model_filename)

    # instantiate LEMS lib
    model = Model()
    model.import_from_file(fp_xml)

    # do some inventory. check if boundaries are set for any sv to print the boundaries section in template
    svboundaries = 0
    for i, sv in enumerate(model.component_types[model_filename].dynamics.state_variables):
        if sv.boundaries != 'None' and sv.boundaries != '' and sv.boundaries:
            svboundaries = 1
            continue

    return model, svboundaries


def default_template():
    here = os.path.dirname(os.path.abspath(__file__))
    tmp_filename = os.path.join(here, 'tmpl8_regTVB.py')
    template = Template(filename=tmp_filename)
    return template


def render_model(model_name, template=None):
    model, svboundaries = load_model(model_name)
    template = template or default_template()
    model_str = template.render(
                            dfunname=model_name,
                            const=model.component_types[model_name].constants,
                            dynamics=model.component_types[model_name].dynamics,
                            svboundaries=svboundaries,
                            exposures=model.component_types[model_name].exposures
                            )
    # print(model_str)
    return model_str


def regTVB_templating(model_filename):
    """
    modelfile.py is placed results into tvb/simulator/models
    for new models models/__init.py__ is auto_updated if model is unfamiliar to tvb
    file_class_name is the name of the produced file and also the class name

    """

    # file locations
    modelfile = "{}{}{}{}".format(os.path.dirname(tvb.__file__),'/simulator/models/',model_filename.lower(),'.py')

    # start templating
    model_str = render_model(model_filename, svboundaries, template=default_template())

    # write templated model to file
    with open(modelfile, "w") as f:
        f.writelines(model_str)

    # write new model to init.py such it is familiar to TVB if not already present
    try:
        doprint=True
        modelenumnum=0
        modulemodnum=0
        with open("{}{}".format(os.path.dirname(tvb.__file__),'/simulator/models/__init__.py'), "r+") as f:
            lines = f.readlines()
            for num, line in enumerate(lines):
                if (model_filename.upper() + ' = ' + "\"" + model_filename + "\"") in line:
                    doprint=False
                elif ("class ModelsEnum(Enum):") in line:
                    modelenumnum = num
                elif ("_module_models = {") in line:
                    modulemodnum = num
            if doprint:
                lines.insert(modelenumnum + 1, "    " + model_filename.upper() + ' = ' + "\"" + model_filename + "\"\n")
                lines.insert(modulemodnum + 2, "    " + "'" + model_filename.lower() + "'" + ': '
                             + "[ModelsEnum." + model_filename.upper() + "],\n")
                f.truncate(0)
                f.seek(0)
                f.writelines(lines)
    except:
        print('unable to add new model to __init__.py')