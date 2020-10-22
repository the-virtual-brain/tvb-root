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
import re
# from tvb.basic.logger.builder import get_logger

# not ideal but avoids modifying  the vendored LEMS itself
sys.path.append(os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
from lems.model.model import Model

# logger = get_logger(__name__)

class rateml:

    def __init__(self, model_filename, language='python', XMLfolder=None, GENfolder=None):
        self.model_filename = model_filename
        self.language = language
        self.XMLfolder = XMLfolder
        self.GENfolder = GENfolder

        # set file locations
        self.generated_model_location = self.set_generated_model_location()
        self.xml_location = self.set_XML_model_folder()

    def default_XML_folder(self):
        here = os.path.dirname(os.path.abspath(__file__))
        xmlpath = os.path.join(here, 'XMLmodels')
        return xmlpath

    def set_XML_model_folder(self):
        folder = self.XMLfolder or self.default_XML_folder()
        return os.path.join(folder, self.model_filename.lower() + '.xml')

    def default_generation_folder(self):
        here = os.path.dirname(os.path.abspath(__file__))
        xmlpath = os.path.join(here, 'generatedModels')
        return xmlpath

    def set_generated_model_location(self):
        folder = self.GENfolder or self.default_generation_folder()
        lan = self.language.lower()
        if lan=='python':
            ext='.py'
        elif lan=='cuda':
            ext='.c'
        return os.path.join(folder, self.model_filename.lower() + ext)

    def model_template(self):
        here = os.path.dirname(os.path.abspath(__file__))
        tmp_filename = os.path.join(here, 'tmpl8_'+ self.language +'.py')
        template = Template(filename=tmp_filename)
        return template

    def XSD_validate_XML(self, file_name):

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

    def inventorize_props(self, model):

        ''' Do some preprocessing on the template to easify rendering '''

        # check if boundaries for state variables are present. contruct is not necessary in pymodels
        # python only
        svboundaries = False
        for i, sv in enumerate(model.component_types['derivatives'].dynamics.state_variables):
            if sv.exposure != 'None' and sv.exposure != '' and sv.exposure:
                svboundaries = True
                continue

        # check for component_types containing coupling in name and gather data.
        # multiple coupling functions could be defined in xml
        # cuda only
        couplinglist = list()
        for i, cplists in enumerate(model.component_types):
            if 'coupling' in cplists.name:
                couplinglist.append(cplists)

        # only check whether noise is there, if so then activate it
        # cuda only
        noisepresent=False
        for ct in (model.component_types):
            if ct.name == 'noise' and ct.description == 'on':
                noisepresent=True

        # see if nsig derived parameter is present for noise
        # cuda only
        nsigpresent=False
        if noisepresent==True:
            for dprm in (modellist.derived_parameters):
                if (dprm.name == 'nsig' or dprm.name == 'NSIG'):
                     nsigpresent=True

        # check for power symbol and translate to python (**) or c power (powf(x, y))
        # there are 5 locations where they can occur: Derivedvariable.value, ConditionalDerivedVariable.Case.condition
        # Derivedparameter.value, Time_Derivaties.value and Exposure.name
        # Todo make more generic
        powlst = model.component_types['derivatives']
        power_parse_exprs_value = [powlst.derived_parameters, powlst.dynamics.derived_variables,
                                   powlst.dynamics.time_derivatives]

        for x, pwr_parse_object in enumerate(power_parse_exprs_value):
            for pwr_obj in pwr_parse_object:
                if '^' in  pwr_obj.value:
                    if self.language=='python':
                        pwr_parse_object[pwr_obj.name].value = pwr_obj.value.replace('{', '')
                        pwr_parse_object[pwr_obj.name].value = pwr_obj.value.replace('^', ' ** ')
                        pwr_parse_object[pwr_obj.name].value = pwr_obj.value.replace('}', '')
                    if self.language=='cuda':
                        for power in re.finditer('\{(.*?)\}',  pwr_obj.value):
                            target = power.group(1)
                            powersplit = target.split('^')
                            powf = 'powf(' + powersplit[0] + ', ' + powersplit[1] + ')'
                            pwr_parse_object[pwr_obj.name].value = pwr_obj.value.replace(target, powf)

        for pwr_obj in powlst.exposures:
            if '^' in pwr_obj.dimension:
                if self.language=='python':
                    powlst.exposures[pwr_obj.name].dimension = pwr_obj.dimension.replace('{', '')
                    powlst.exposures[pwr_obj.name].dimension = pwr_obj.dimension.replace('^', ' ** ')
                    powlst.exposures[pwr_obj.name].dimension = pwr_obj.dimension.replace('}', '')
                if self.language=='cuda':
                    for power in re.finditer('\{(.*?)\}', pwr_obj.dimension):
                        target = power.group(1)
                        powersplit = target.split('^')
                        powf = 'powf(' + powersplit[0] + ', ' + powersplit[1] + ')'
                        powlst.exposures[pwr_obj.name].dimension = pwr_obj.dimension.replace(target, powf)

        for cdv in powlst.dynamics.conditional_derived_variables:
            for casenr, case in enumerate(cdv.cases):
                if '^' in case.value:
                    if self.language == 'python':
                        powlst.dynamics.conditional_derived_variables[cdv.name].cases[casenr].value = case.value.replace('{', '')
                        powlst.dynamics.conditional_derived_variables[cdv.name].cases[casenr].value = case.value.replace('^', ' ** ')
                        powlst.dynamics.conditional_derived_variables[cdv.name].cases[casenr].value = case.value.replace('}', '')
                    if self.language == 'cuda':
                        for power in re.finditer('\{(.*?)\}', case.value):
                            target = power.group(1)
                            powersplit = target.split('^')
                            powf = 'powf(' + powersplit[0] + ', ' + powersplit[1] + ')'
                            powlst.dynamics.conditional_derived_variables[cdv.name].cases[casenr].value = case.value.replace(target, powf)

        # print((powlst.derived_parameters['rec_speed_dt'].value))
        # print((powlst.dynamics.derived_variables['bla'].value))
        # print((powlst.exposures['x1'].dimension))
        # print((powlst.dynamics.conditional_derived_variables['ydot0'].cases[0].value))
        # print((powlst.dynamics.conditional_derived_variables['ydot0'].cases[1].value))
        # print((powlst.dynamics.conditional_derived_variables['ydot2'].cases[0].value))

        return svboundaries, couplinglist, noisepresent, nsigpresent

    def load_model(self):
        "Load model from filename"

        # instantiate LEMS lib
        model = Model()
        model.import_from_file(self.xml_location)

        self.XSD_validate_XML(self.xml_location)

        # do some inventory. check if boundaries are set for any sv to print the boundaries section in template
        svboundaries, couplinglist, noisepresent, nsigpresent = self.inventorize_props(model)

        return model, svboundaries, couplinglist, noisepresent, nsigpresent

    def render_model(self):
        '''
        render_model start the mako templating.
        this function is similar for all languages. its .render arguments are overloaded.
        '''

        model, svboundaries, couplinglist, noisepresent, nsigpresent = self.load_model()

        derivative_list = model.component_types['derivatives']

        # start templating
        model_str = self.model_template().render(
            modelname=self.model_filename,                  # all
            const=derivative_list.constants,                # all
            dynamics=derivative_list.dynamics,              # all
            exposures=derivative_list.exposures,            # all
            params=derivative_list.parameters,              # cuda
            derparams=derivative_list.derived_parameters,   # cuda
            svboundaries=svboundaries,                      # python
            coupling=couplinglist,                          # cuda
            noisepresent=noisepresent,                      # cuda
            nsigpresent=nsigpresent,                        # cuda
            )

        return model_str

    def familiarize_TVB(self, model_filename, model_str):
        '''
        Write new model to TVB model location and into init.py such it is familiar to TVB if not already present
        This is for Python models only
        '''

        # set tvb location
        TVB_model_location = os.path.join(os.path.dirname(tvb.simulator.models.__file__), model_filename.lower() + 'T.py')
        # next to user submitted location also write to default tvb location
        self.write_model_file(TVB_model_location, model_str)

        try:
            doprint = True
            modelenumnum = 0
            modulemodnum = 0
            with open(os.path.join(os.path.dirname(tvb.simulator.models.__file__), '__init__.py'), "r+") as f:
                lines = f.readlines()
                for num, line in enumerate(lines):
                    if (model_filename.upper() + 'T = ' + "\"" + model_filename + "T\"") in line:
                        doprint = False
                    elif ("class ModelsEnum(Enum):") in line:
                        modelenumnum = num
                    elif ("_module_models = {") in line:
                        modulemodnum = num
                if doprint:
                    lines.insert(modelenumnum + 1, "    " + model_filename.upper() + 'T = ' + "\"" + model_filename + "T\"\n")
                    lines.insert(modulemodnum + 2, "    " + "'" + model_filename.lower() + "T'" + ': '
                                 + "[ModelsEnum." + model_filename.upper() + "T],\n")
                    f.truncate(0)
                    f.seek(0)
                    f.writelines(lines)
                # logger.info("model file generated {}".format(modelfile))
        except IOError as e:
            print(e)
            # logger.error('ioerror: %s', e)

    def write_model_file(self, model_location, model_str):

        '''Write templated model to file'''

        with open(model_location, "w") as f:
            f.writelines(model_str)

    def start_model_generation(self):

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

        # start templating
        model_str = self.render_model()

        # write model to user submitted location
        self.write_model_file(self.generated_model_location, model_str)

        # if it is a TVB.py model, it should be familiarized
        if self.language.lower()=='python':
            self.familiarize_TVB(self.model_filename, model_str)

if __name__ == "__main__":
    # model_filename = 'MontbrioT'
    # model_filename = 'Generic2dOscillatorT'
    # model_filename = 'KuramotoT'
    # model_filename = 'EpileptorT'
    # model_filename = 'ReducedWongWangT'
    model_filename = 'epileptor'

    language='python'
    # language='cuda'
    rateml(model_filename, language, './XMLmodels/', './generatedModels/').start_model_generation()
