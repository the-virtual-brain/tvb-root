# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2017, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
.. moduleauthor:: Ionel Ortelecan <ionel.ortelecan@codemart.ro>
"""

import numpy
from copy import deepcopy
from tvb.basic.traits.core import Type
from tvb.core.adapters.abcadapter import KEY_EQUATION, KEY_FOCAL_POINTS, KEY_SURFACE_GID, ABCAdapter
from tvb.datatypes.equations import SpatialApplicableEquation, Gaussian
from tvb.basic.logger.builder import get_logger
from tvb.simulator.models import Generic2dOscillator


KEY_FOCAL_POINTS_TRIANGLES = "focal_points_triangles"


class SurfaceContextModelParameters(object):
    """
    This class contains methods which allows you to edit the model
    parameters for each vertex of the given surface.
    """
    def __init__(self, surface, default_model=None):
        self.logger = get_logger(self.__class__.__module__)

        if default_model is not None:
            self.default_model = default_model
        else:
            self.default_model = Generic2dOscillator()

        self.model_name = self.default_model.__class__.__name__
        self.model_parameter_names = self.default_model.ui_configurable_parameters[:]

        if not self.model_parameter_names:
            self.logger.warning("The 'ui_configurable_parameters' list of the current model is empty!")

        self.prepared_model_parameter_names = self._prepare_parameter_names(self.model_parameter_names)
        self.surface = surface
        self.applied_equations = {}


    @staticmethod
    def _prepare_parameter_names(parameter_names):
        """
        Used for removing the '_' character from the parameter_names.
        """
        #{$orig_param: $new_param}
        result = {}
        for param in parameter_names:
            result[param] = param.replace("_", "")
        return result


    def apply_equation(self, param_name, equation_instance):
        """
        Applies an equation on the given model parameter.
        """
        if param_name in self.applied_equations:
            param_data = self.applied_equations[param_name]
            param_data[KEY_EQUATION] = equation_instance
        else:
            self.applied_equations[param_name] = {KEY_EQUATION: equation_instance, KEY_FOCAL_POINTS: [],
                                                  KEY_SURFACE_GID: self.surface.gid, KEY_FOCAL_POINTS_TRIANGLES: [],
                                                  ABCAdapter.KEY_DTYPE: equation_instance.__class__.__module__ + '.' +
                                                                        equation_instance.__class__.__name__}


    def apply_focal_point(self, model_param, triangle_index):
        """
        NOTE: Expects a triangle index

        Adds a focal point in which should be applied the equation for the given model parameter.
        """
        triangle_index = int(triangle_index)
        if model_param in self.applied_equations:
            if triangle_index >= 0:
                vertex_index = int(self.surface.triangles[triangle_index][0])
                model_equation = self.applied_equations[model_param]
                if vertex_index not in model_equation[KEY_FOCAL_POINTS]:
                    model_equation[KEY_FOCAL_POINTS].append(vertex_index)
                    model_equation[KEY_FOCAL_POINTS_TRIANGLES].append(triangle_index)


    def remove_focal_point(self, model_param, triangle_index):
        """
        NOTE: Expects a vertex index

        Removes a focal point from the list of focal points in which should be
        applied the equation for the given model parameter.
        """
        triangle_index = int(triangle_index)
        if model_param in self.applied_equations:
            if triangle_index >= 0:
                model_equation = self.applied_equations[model_param]

                if triangle_index in model_equation[KEY_FOCAL_POINTS_TRIANGLES]:
                    f_p_idx = model_equation[KEY_FOCAL_POINTS_TRIANGLES].index(triangle_index)
                    model_equation[KEY_FOCAL_POINTS].remove(model_equation[KEY_FOCAL_POINTS][f_p_idx])
                    model_equation[KEY_FOCAL_POINTS_TRIANGLES].remove(triangle_index)


    def reset_equations_for_all_parameters(self):
        """
        Reset the equations for all the model parameters.
        """
        self.applied_equations = {}


    def reset_param_equation(self, model_param):
        """
        Resets the equation for the specified model parameter.
        """
        if model_param in self.applied_equations:
            self.applied_equations.pop(model_param)


    def get_equation_for_parameter(self, parameter_name):
        """
        :returns: the applied equation for the given model param OR None if there is no equation applied to this param.
        """
        try:
            return self.applied_equations[parameter_name][KEY_EQUATION]
        except KeyError:
            return None


    def get_focal_points_for_parameter(self, parameter_name):
        """
        :returns: the list of focal points for the equation applied in the given model param.
        """
        if parameter_name in self.applied_equations and KEY_FOCAL_POINTS in self.applied_equations[parameter_name]:
            # todo did the above check intent to be for KEY_FOCAL_POINTS_TRIANGLES?
            return self.applied_equations[parameter_name][KEY_FOCAL_POINTS_TRIANGLES]
        return []


    def get_data_for_model_param(self, original_param_name, modified_param_name):
        """
        :returns: a dictionary of form {"equation": $equation, "focal_points": $list_of_focal_points,
                    "no_of_vertices": $surface_no_of_vertices} if the user specified any equation for computing
                    the value of the given parameter, OR a string of form: "[$default_model_param_value]"
                    if the user didn't specified an equation for the given param
        """
        if modified_param_name in self.applied_equations:
            return self.applied_equations[modified_param_name]
        else:
            default_attr = deepcopy(getattr(self.default_model, original_param_name))
            if isinstance(default_attr, numpy.ndarray):
                default_attr = default_attr.tolist()
                return str([default_attr[0]])
            return str([default_attr])


    def get_configure_info(self):
        """
        :returns: a dictionary which contains information about the applied equations on the model parameters.
        """
        result = {}
        for param in self.applied_equations:
            equation = self.applied_equations[param][KEY_EQUATION]
            keys = sorted(equation.parameters.keys(), key=lambda x: len(x))
            keys.reverse()
            base_equation = equation.trait['equation'].interface['description']
            for eq_param in keys:
                while True:
                    stripped_eq = "".join(base_equation.split())
                    param_idx = stripped_eq.find('\\' + eq_param)
                    if param_idx < 0:
                        break
                    #If parameter is precedeed by an alfanumerical character replace with multiplicative sign
                    if param_idx > 0 and stripped_eq[param_idx - 1].isalnum():
                        base_equation = base_equation.replace('\\' + eq_param, '*' + str(equation.parameters[eq_param]))
                    else:
                        base_equation = base_equation.replace('\\' + eq_param, str(equation.parameters[eq_param]))
                base_equation = base_equation.replace(eq_param, str(equation.parameters[eq_param]))
            focal_points = str(self.applied_equations[param][KEY_FOCAL_POINTS])
            result[param] = {'equation_name': equation.__class__.__name__,
                             'equation_params': base_equation, 'focal_points': focal_points}
        return result



class EquationDisplayer(Type):
    """
    Class used for generating the UI related to equations.
    """
    model_param_equation = SpatialApplicableEquation(label='Equation', default=Gaussian)
    
    