# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2020, Baycrest Centre for Geriatric Care ("Baycrest") and others
#
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this
# program.  If not, see <http://www.gnu.org/licenses/>.
#
#
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

"""
LEMS2python module implements a DSL code generation using a TVB-specific LEMS-based DSL.

.. moduleauthor:: Michiel. A. van der Vlag <m.van.der.vlag@fz-juelich.de>   
.. moduleauthor:: Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>

"""

import os, sys
import tvb.simulator.models
from mako.template import Template
# from tvb.basic.logger.builder import get_logger

# not ideal but avoids modifying  the vendored LEMS itself
sys.path.append(os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
from lems.model.model import Model


# logger = get_logger(__name__)


def default_lems_folder():
    here = os.path.dirname(os.path.abspath(__file__))
    xmlpath = os.path.join(here, 'XMLmodels')
    return xmlpath


def lems_file(model_name, folder=None):
    folder = folder or default_lems_folder()
    return os.path.join(folder, model_name.lower() + '.xml')


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

    # instantiate LEMS lib
    model = Model()
    model.import_from_file(fp_xml)

    XSD_validate_XML(fp_xml)
    # do some inventory. check if boundaries are set for any sv to print the boundaries section in template
    svboundaries = 0
    for i, sv in enumerate(model.component_types['derivatives'].dynamics.state_variables):
        if sv.exposure != 'None' and sv.exposure != '' and sv.exposure:
            svboundaries = 1
            continue

    return model, svboundaries


def default_template():
    here = os.path.dirname(os.path.abspath(__file__))
    tmp_filename = os.path.join(here, 'tmpl8_regTVB.py')
    template = Template(filename=tmp_filename)
    return template


def render_model(model_name, template=None, folder=None):
    model, svboundaries = load_model(model_name, folder)
    template = template or default_template()

    modellist = model.component_types['derivatives']

    model_str = template.render(
        dfunname=model_name,
        const=modellist.constants,
        dynamics=modellist.dynamics,
        svboundaries=svboundaries,
        exposures=modellist.exposures
    )
    return model_str


def regTVB_templating(model_filename, folder=None):
    """
    modelfile.py is placed results into tvb/simulator/models
    for new models models/__init.py__ is auto_updated if model is unfamiliar to tvb
    file_class_name is the name of the produced file and also the model's class name
    the path to XML model files folder can be added with the 2nd argument.
    example model files:
        epileptort.xml
        generic2doscillatort.xml
        kuramotot.xml
        montbriot.xml
        reducedwongwangt.xml
    """

    # file locations
    modelfile = os.path.join(os.path.dirname(tvb.simulator.models.__file__), model_filename.lower() + '.py')

    # start templating
    model_str = render_model(model_filename, template=default_template(), folder=folder)

    # write templated model to file
    with open(modelfile, "w") as f:
        f.writelines(model_str)

    # write new model to init.py such it is familiar to TVB if not already present
    try:
        doprint = True
        modelenumnum = 0
        modulemodnum = 0
        with open(os.path.join(os.path.dirname(tvb.simulator.models.__file__), '__init__.py'), "r+") as f:
            lines = f.readlines()
            for num, line in enumerate(lines):
                if (model_filename.upper() + ' = ' + "\"" + model_filename + "\"") in line:
                    doprint = False
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
            # logger.info("model file generated {}".format(modelfile))
    except IOError as e:
        print(e)
        # logger.error('ioerror: %s', e)


if __name__ == "__main__":
    # model_filename = 'MontbrioT'
    # model_filename = 'Generic2dOscillatorT'
    # model_filename = 'KuramotoT'
    # model_filename = 'EpileptorT'
    model_filename = 'ReducedWongWangT'
    regTVB_templating(model_filename, './XMLmodels/')
