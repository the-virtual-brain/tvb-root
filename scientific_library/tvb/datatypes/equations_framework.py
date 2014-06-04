# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and 
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2013, Baycrest Centre for Geriatric Care ("Baycrest")
#
# This program is free software; you can redistribute it and/or modify it under 
# the terms of the GNU General Public License version 2 as published by the Free
# Software Foundation. This program is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details. You should have received a copy of the GNU General 
# Public License along with this program; if not, you can download it here
# http://www.gnu.org/licenses/old-licenses/gpl-2.0
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

Framework methods for the Equation datatypes.

.. moduleauthor:: Ionel Ortelecan <ionel.ortelecan@codemart.ro>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>

"""
import json
import numpy
from tvb.datatypes import equations_data
from tvb.basic.traits import parameters_factory
from tvb.basic.logger.builder import get_logger

LOG = get_logger(__name__)

# In how many points should the equation be evaluated for the plot. Increasing this will
# give smoother results at the cost of some performance
DEFAULT_PLOT_GRANULARITY = 1024


class EquationFramework(equations_data.EquationData):
    """ This class exists to add framework methods to EquationData. """
    __tablename__ = None

    def get_series_data(self, min_range=0, max_range=100, step=None):
        """
        NOTE: The symbol from the equation which varies should be named: var
        Returns the series data needed for plotting this equation.
        """
        if step is None:
            step = float(max_range - min_range) / DEFAULT_PLOT_GRANULARITY

        var = numpy.arange(min_range, max_range+step, step)
        var = var[numpy.newaxis, :]

        self.pattern = var
        y = self.pattern
        result = zip(var.flat, y.flat)
        return result, False


    @staticmethod
    def build_equation_from_dict(equation_field_name, submitted_data_dict, alter_submitted_dictionary=False):
        """
        Builds from the given data dictionary the equation for the specified field name.
        The dictionary should have the data collapsed.
        """
        if equation_field_name not in submitted_data_dict:
            return None

        eq_param_str = equation_field_name + '_parameters'
        eq = submitted_data_dict.get(eq_param_str)

        equation_parameters = {}
        if eq:
            if 'parameters' in eq:
                equation_parameters = eq['parameters']
            if 'parameters_parameters' in eq:
                equation_parameters = eq['parameters_parameters']

        for k in equation_parameters:
            equation_parameters[k] = float(equation_parameters[k])

        equation_type = submitted_data_dict[equation_field_name]
        equation = parameters_factory.get_traited_instance_for_name(equation_type, equations_data.EquationData,
                                                                    {'parameters': equation_parameters})
        if alter_submitted_dictionary:
            del submitted_data_dict[eq_param_str]
            submitted_data_dict[equation_field_name] = equation

        return equation


    @staticmethod
    def to_json(entity):
        """
        Returns the json representation of this equation.

        The representation of an equation is a dictionary with the following form:
        {'equation_type': '$equation_type', 'parameters': {'$param_name': '$param_value', ...}}
        """
        if entity is not None:
            result = {'__mapped_module': entity.__class__.__module__,
                      '__mapped_class': entity.__class__.__name__,
                      'parameters': entity.parameters}
            return json.dumps(result)
        return None


    @staticmethod
    def from_json(string):
        """
        Retrieves an instance to an equation represented as JSON.

        :param string: the JSON representation of the equation
        :returns: a `tvb.datatypes.equations_data` equation instance
        """
        loaded_dict = json.loads(string)
        if loaded_dict is None:
            return None
        modulename = loaded_dict['__mapped_module']
        classname = loaded_dict['__mapped_class']
        module_entity = __import__(modulename, globals(), locals(), [classname])
        class_entity = getattr(module_entity, classname)
        loaded_instance = class_entity()
        loaded_instance.parameters = loaded_dict['parameters']
        return loaded_instance


class DiscreteEquationFramework(equations_data.DiscreteEquationData, EquationFramework):
    """ This class exists to add framework methods to DiscreteData """


class LinearFramework(equations_data.LinearData, EquationFramework):
    """ This class exists to add framework methods to LinearData """


class GaussianFramework(equations_data.GaussianData, EquationFramework):
    """ This class exists to add framework methods to GaussianData """


class DoubleGaussianFramework(equations_data.DoubleGaussianData, EquationFramework):
    """ This class exists to add framework methods to DoubleGaussianData """


class SigmoidFramework(equations_data.SigmoidData, EquationFramework):
    """ This class exists to add framework methods to SigmoidData """


class GeneralizedSigmoidFramework(equations_data.GeneralizedSigmoidData, EquationFramework):
    """ This class exists to add framework methods to GeneralizedSigmoidData """


class SinusoidFramework(equations_data.SinusoidData, EquationFramework):
    """ This class exists to add framework methods to SinusoidData """


class CosineFramework(equations_data.CosineData, EquationFramework):
    """ This class exists to add framework methods to CosineData """


class AlphaFramework(equations_data.AlphaData, EquationFramework):
    """ This class exists to add framework methods to AlphaData """


class PulseTrainFramework(equations_data.PulseTrainData, EquationFramework):
    """ This class exists to add framework methods to PulseTrainData """


class GammaFramework(equations_data.GammaData, EquationFramework):
    """ This class exists to add framework methods to GammaData """


class DoubleExponentialFramework(equations_data.DoubleExponentialData, EquationFramework):
    """ This class exists to add framework methods to GammaData """


class FirstOrderVolterraFramework(equations_data.DoubleExponentialData, EquationFramework):
    """ This class exists to add framework methods to GammaData """


class MixtureOfGammasFramework(equations_data.MixtureOfGammasData, EquationFramework):
    """ This class exists to add framework methods to GammaData """
