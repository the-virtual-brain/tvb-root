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


        Usage: - create an modelfile.xml
               - import rateML
               - make instance: rateml(model_filename, language, your/XMLfolder, 'your/generatedModelsfolder')
        the current supported model framework languages are python and cuda.
        for new models models/__init.py__ is auto_updated if model is unfamiliar to tvb for py models
        model_class_name is the model_filename + 'T', so not to overwrite existing models. Be sure to add the t
        when simulating the model in TVB
        example model files:
            epileptor.xml
            generic2doscillatort.xml
            kuramoto.xml
            montbrio.xml
            reducedwongwang.xml

"""

import os
import re
import argparse
import numpy as np
import tvb.simulator.models
from mako.template import Template
from tvb.basic.logger.builder import get_logger
from lems.model.model import Model

logger = get_logger(__name__)


class RateML:

    def __init__(self, model_filename=None, language=None, XMLfolder=None, GENfolder=None):

        self.args = self.parse_args()

        self.model_filename = model_filename or self.args.model

        language = language or self.args.language.lower()

        try:
            assert language in ('cuda', 'python', 'Cuda', 'Python', 'CUDA')
            self.language = language.lower()
        except AssertionError as e:
            logger.error('Please choose between Python or Cuda %s', e)
            exit()

        self.XMLfolder = XMLfolder or self.args.source
        self.GENfolder = GENfolder or self.args.destination

        # set file locations
        self.generated_model_location = self.set_generated_model_location()
        self.xml_location = self.set_XML_model_folder()

        # start templating
        model_str, driver_str = self.render()

        # write model to user submitted location
        self.write_model_file(self.generated_model_location, model_str)

        if self.language == 'cuda':
            # write driver file to fixed ./run/ location
            self.write_model_file(self.set_driver_location(), driver_str)
            # for driver robustness also write XML to default location generatedModels folder for CUDA
            if GENfolder != None:
                default_save = os.path.join(self.default_generation_folder(), self.model_filename.lower() + '.c')
                self.write_model_file(default_save, model_str)

        # if it is a TVB.py model, it should be familiarized
        if self.language.lower() == 'python':
            self.familiarize_TVB(model_str)

    def parse_args(self):  # {{{
        parser = argparse.ArgumentParser(description='Run XML model conversion.')

        parser.add_argument('-m', '--model', default='rwongwang',
                            help="neural mass model to be converted")
        parser.add_argument('-l', '--language', default='Python',
                            help="programming language output")
        parser.add_argument('-s', '--source', default='',
                            help="source folder location")
        parser.add_argument('-d', '--destination', default='',
                            help="destination folder location")

        args = parser.parse_args()
        return args

    @staticmethod
    def default_XML_folder():
        here = os.path.dirname(os.path.abspath(__file__))
        xmlpath = os.path.join(here, 'XMLmodels')
        return xmlpath

    def set_XML_model_folder(self):
        folder = self.XMLfolder or self.default_XML_folder()

        try:
            location = os.path.join(folder, self.model_filename.lower() + '.xml')
            assert os.path.isfile(location)
        except AssertionError:
            logger.error('XML folder %s does not contain %s', folder, self.model_filename + '.xml')
            exit()

        return location

    @staticmethod
    def default_generation_folder():
        here = os.path.dirname(os.path.abspath(__file__))
        modelpath = os.path.join(here, 'generatedModels')
        return modelpath

    def set_generated_model_location(self):
        folder = self.GENfolder or self.default_generation_folder()
        lan = self.language.lower()
        if lan == 'python':
            ext = '.py'
        elif lan == 'cuda':
            ext = '.c'

        try:
            location = os.path.join(folder)
            assert os.path.isdir(location)
        except AssertionError:
            logger.error('Generation folder %s does not exist', location)
            exit()

        return os.path.join(folder, self.model_filename.lower() + ext)

    def set_driver_location(self):
        here = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(here, 'run', 'model_driver_' + self.model_filename + '.py')

    def set_template(self, name):
        here = os.path.dirname(os.path.abspath(__file__))
        tmp_filename = os.path.join(here, 'tmpl8_' + name + '.py')
        template = Template(filename=tmp_filename)
        return template

    def XSD_validate_XML(self):
        """Use own validation instead of LEMS because of slight difference in definition file"""
        from lxml import etree

        xsd_fname = os.path.join(os.path.abspath(os.path.dirname(__file__)), "rML_v0.xsd")
        xmlschema = etree.XMLSchema(etree.parse(xsd_fname))
        xmlschema.assertValid(etree.parse(self.xml_location))
        logger.info("True validation of {0} against {1}".format(self.xml_location, xsd_fname))

    def pp_bound(self, model):

        # check if boundaries for state variables are present. contruct is not necessary in pymodels
        # python only
        svboundaries = False
        for i, sv in enumerate(model.component_types['derivatives'].dynamics.state_variables):
            if sv.exposure != 'None' and sv.exposure != '' and sv.exposure:
                svboundaries = True
                continue

        return svboundaries

    def pp_cplist(self, model):

        # check for component_types containing coupling in name and gather data.
        # multiple coupling functions could be defined in xml
        # cuda only
        couplinglist = list()
        for i, cplists in enumerate(model.component_types):
            if 'coupling' in cplists.name:
                couplinglist.append(cplists)

        return couplinglist

    def pp_noise(self, model):

        # only check whether noise is there, if so then activate it
        # cuda only
        noisepresent = False
        for ct in (model.component_types):
            if ct.name == 'noise':
                noisepresent = True

        # see if nsig derived parameter is present for noise
        # cuda only
        modellist = model.component_types['derivatives']
        nsigpresent = False
        if noisepresent == True:
            for dprm in (modellist.derived_parameters):
                if (dprm.name == 'nsig' or dprm.name == 'NSIG'):
                    nsigpresent = True

        return noisepresent, nsigpresent

    # check for power symbol and parse to python (**) or c power (powf(x, y))
    def swap_language_specific_terms(self, model_str):

        if self.language == 'cuda':
            model_str = re.sub(r"\bpi\b", 'PI', model_str)
            model_str = re.sub(r"\binf\b", 'INF', model_str)

        for power in re.finditer(r"\{(.*?)(\^)(.*?)\}", model_str):
            target = power.group(0)
            powersplit = target.split('^')
            if self.language == 'cuda':
                pow = 'powf(' + powersplit[0].replace('{', '') + ', ' + powersplit[1].replace('}', '') + ')'
            if self.language == 'python':
                pow = powersplit[0].replace('{', '') + '**' + powersplit[1].replace('}', '')
            model_str = re.sub(re.escape(target), pow, model_str)

        return model_str

    # setting the inital value for cuda models
    # the entered range is splitted and a random value is generated within range
    # if values are equal then that is the inital value
    def init_statevariables(self, model):

        modellist = model.component_types['derivatives'].dynamics.state_variables
        for sv in modellist:
            splitdim = list(sv.dimension.split(","))
            sv_rnd = np.random.uniform(low=float(splitdim[0]), high=float(splitdim[1]))
            model.component_types['derivatives'].dynamics.state_variables[sv.name].dimension = sv_rnd

    def load_model(self):
        """Load model from filename"""

        # instantiate LEMS lib
        model = Model()
        model.import_from_file(self.xml_location)

        self.XSD_validate_XML()

        # Do some preprocessing on the template to easify rendering
        noisepresent, nsigpresent = self.pp_noise(model)
        couplinglist = self.pp_cplist(model)
        svboundaries = self.pp_bound(model)

        if self.language == 'cuda':
            self.init_statevariables(model)

        return model, svboundaries, couplinglist, noisepresent, nsigpresent

    def render(self):
        """
        render_model start the mako templating.
        this function is similar for all languages. its .render arguments are overloaded.
        """

        model, svboundaries, couplinglist, noisepresent, nsigpresent = self.load_model()

        derivative_list = model.component_types['derivatives']

        model_str = self.render_model(derivative_list, svboundaries, couplinglist, noisepresent, nsigpresent)

        model_str = self.swap_language_specific_terms(model_str)

        # render driver only in case of cuda
        if self.language == 'cuda':
            driver_str = self.render_driver(derivative_list)
        else:
            driver_str = None

        return model_str, driver_str

    def render_model(self, derivative_list, svboundaries, couplinglist, noisepresent, nsigpresent):

        if self.language == 'python':
            model_class_name = self.model_filename.capitalize() + 'T'
        if self.language == 'cuda':
            model_class_name = self.model_filename

        # start templating
        model_str = self.set_template(self.language).render(
            modelname=model_class_name,  # all
            const=derivative_list.constants,  # all
            dynamics=derivative_list.dynamics,  # all
            exposures=derivative_list.exposures,  # all
            params=derivative_list.parameters,  # cuda
            derparams=derivative_list.derived_parameters,  # cuda
            svboundaries=svboundaries,  # python
            coupling=couplinglist,  # cuda
            noisepresent=noisepresent,  # cuda
            nsigpresent=nsigpresent,  # cuda
        )

        return model_str

    def render_driver(self, derivative_list):

        driver_str = self.set_template('driver').render(
            model=self.model_filename,
            XML=derivative_list,
        )

        return driver_str

    def familiarize_TVB(self, model_str):
        '''
        Write new model to TVB model location and into init.py such it is familiar to TVB if not already present
        This is for Python models only
        '''

        model_filename = self.model_filename
        # set tvb location
        TVB_model_location = os.path.join(os.path.dirname(tvb.simulator.models.__file__),
                                          model_filename.lower() + 'T.py')
        # next to user submitted location also write to default tvb location
        self.write_model_file(TVB_model_location, model_str)

        try:
            doprint = True
            modelenumnum = 0
            modulemodnum = 0
            with open(os.path.join(os.path.dirname(tvb.simulator.models.__file__), '__init__.py'), "r+") as f:
                lines = f.readlines()
                for num, line in enumerate(lines):
                    if (model_filename.upper() + 'T = ' + "\"" + model_filename.capitalize() + "T\"") in line:
                        doprint = False
                    elif ("class ModelsEnum(Enum):") in line:
                        modelenumnum = num
                    elif ("_module_models = {") in line:
                        modulemodnum = num
                if doprint:
                    lines.insert(modelenumnum + 1, "    " + model_filename.upper() + 'T = ' + "\"" +
                                 model_filename.capitalize() + "T\"\n")
                    lines.insert(modulemodnum + 2, "    " + "'" + model_filename.lower() + "T'" + ': '
                                 + "[ModelsEnum." + model_filename.upper() + "T],\n")
                    f.truncate(0)
                    f.seek(0)
                    f.writelines(lines)
                logger.info("model file generated {}".format(model_filename))
        except IOError as e:
            logger.error('Writing TVB model to file failed: %s', e)

    def write_model_file(self, model_location, model_str):

        '''Write templated model to file'''

        try:
            with open(model_location, "w") as f:
                f.writelines(model_str)
        except IOError as e:
            logger.error('Writing %s model to file failed: %s', self.language, e)


if __name__ == "__main__":
    RateML()

    # example for direct project implementation
    # set the language for your model
    # language='python'
    # language='Cuda'

    # choose an example or your own model
    # model_filename = 'montbrio'
    # model_filename = 'oscillator'
    # model_filename = 'kuramoto'
    # model_filename = 'rwongwang'
    # model_filename = 'epileptor'

    # start conversion to default model location
    # RateML(model_filename, language)
