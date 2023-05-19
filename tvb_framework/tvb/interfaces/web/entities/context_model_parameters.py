# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need to download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2023, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
# When using The Virtual Brain for scientific publications, please cite it as explained here:
# https://www.thevirtualbrain.org/tvb/zwei/neuroscience-publications
#
#

"""
.. moduleauthor:: Ionel Ortelecan <ionel.ortelecan@codemart.ro>
"""

from copy import deepcopy

from tvb.adapters.forms.surface_model_parameters_form import SurfaceModelParametersForm
from tvb.basic.logger.builder import get_logger
from tvb.core.neocom import h5
from tvb.core.neotraits.spatial_model import SpatialModel
from tvb.datatypes.surfaces import CorticalSurface

KEY_EQUATION = "equation"
KEY_FOCAL_POINTS = "focal_points"
KEY_FOCAL_POINTS_TRIANGLES = "focal_points_triangles"


class SurfaceContextModelParameters(SpatialModel):
    """
    This class contains methods which allows you to edit the model
    parameters for each vertex of the given surface.
    """

    def __init__(self, surface_index, default_model, current_equation, current_model_param):
        self.logger = get_logger(self.__class__.__module__)

        self.default_model = default_model
        self.surface_index = surface_index
        self.applied_equations = {}
        self.current_equation = current_equation
        self.current_model_param = current_model_param

    def apply_equation(self, param_name, equation_instance):
        """
        Applies an equation on the given model parameter.
        """
        if param_name in self.applied_equations:
            param_data = self.applied_equations[param_name]
            param_data[KEY_EQUATION] = equation_instance
        else:
            self.applied_equations[param_name] = {KEY_EQUATION: equation_instance, KEY_FOCAL_POINTS: [],
                                                  KEY_FOCAL_POINTS_TRIANGLES: []}

    def apply_focal_point(self, model_param, triangle_index):
        """
        NOTE: Expects a triangle index

        Adds a focal point in which should be applied the equation for the given model parameter.
        """
        triangle_index = int(triangle_index)
        surface_h5 = h5.h5_file_for_index(self.surface_index)

        if model_param in self.applied_equations:
            if triangle_index >= 0:
                vertex_index = int(surface_h5.triangles[triangle_index][0])
                model_equation = self.applied_equations[model_param]
                if vertex_index not in model_equation[KEY_FOCAL_POINTS]:
                    model_equation[KEY_FOCAL_POINTS].append(vertex_index)
                    model_equation[KEY_FOCAL_POINTS_TRIANGLES].append(triangle_index)
        surface_h5.close()

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
            return self.applied_equations[parameter_name][KEY_FOCAL_POINTS_TRIANGLES]
        return []

    def _get_default_value_for_model_param(self, param_name):
        default_attr = deepcopy(getattr(type(self.default_model), param_name))
        return default_attr.default

    def get_data_for_model_param(self, param_name):
        """
        Compute the equation configured for the current param_name.
        If no equation was set for param_name, return the default array.
        """
        surface_dt = CorticalSurface()
        surface_h5 = h5.h5_file_for_index(self.surface_index)
        surface_h5.load_into(surface_dt)
        surface_h5.close()

        if param_name in self.applied_equations:
            temp = self.applied_equations[param_name]
            equation = temp[KEY_EQUATION]
            focal_points = temp[KEY_FOCAL_POINTS]
            # if focal points or the equation are missing do not update this model parameter
            if focal_points and equation:
                res = surface_dt.compute_equation(focal_points, equation)
                return res
            self.logger.warning('Focal points or Equation are missing for %s. Defaults will be used.', param_name)
            return self._get_default_value_for_model_param(param_name)
        else:
            return self._get_default_value_for_model_param(param_name)

    def get_configure_info(self):
        """
        :returns: a dictionary which contains information about the applied equations on the model parameters.
        """
        result = {}
        for param in self.applied_equations:
            equation = self.applied_equations[param][KEY_EQUATION]
            keys = sorted(list(equation.parameters), key=lambda x: len(x))
            keys.reverse()
            base_equation = equation.equation
            for eq_param in keys:
                while True:
                    stripped_eq = "".join(base_equation.split())
                    param_idx = stripped_eq.find('\\' + eq_param)
                    if param_idx < 0:
                        break
                    # If parameter is precedeed by an alfanumerical character replace with multiplicative sign
                    if param_idx > 0 and stripped_eq[param_idx - 1].isalnum():
                        base_equation = base_equation.replace('\\' + eq_param, '*' + str(equation.parameters[eq_param]))
                    else:
                        base_equation = base_equation.replace('\\' + eq_param, str(equation.parameters[eq_param]))
                base_equation = base_equation.replace(eq_param, str(equation.parameters[eq_param]))
            focal_points = str(self.applied_equations[param][KEY_FOCAL_POINTS])
            result[param] = {'equation_name': equation.__class__.__name__,
                             'equation_params': base_equation, 'focal_points': focal_points}
        return result

    @staticmethod
    def get_equation_information():
        return {
            SurfaceModelParametersForm.equation_field_label: 'current_equation'
        }
