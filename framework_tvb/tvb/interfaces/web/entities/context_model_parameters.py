# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
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
.. moduleauthor:: Ionel Ortelecan <ionel.ortelecan@codemart.ro>
"""

import numpy
import tvb.basic.traits.parameters_factory as parameters_factory
import tvb.simulator.models as models_module
from copy import deepcopy
from tvb.basic.traits.core import Type
from tvb.core.adapters.abcadapter import KEY_EQUATION, KEY_FOCAL_POINTS, KEY_SURFACE_GID, ABCAdapter
from tvb.datatypes.equations import SpatialApplicableEquation, Gaussian
from tvb.adapters.visualizers.phase_plane_interactive import PhasePlaneInteractive
from tvb.interfaces.web.entities.context_spatial import BaseSpatialContext


KEY_FOCAL_POINTS_TRIANGLES = "focal_points_triangles"



class ContextModelParameters(BaseSpatialContext):
    """
    This class behaves like a controller. The model for this controller will
    be the fields defined into the init method and the view is represented
    by a PhasePlaneInteractive instance.

    This class may also be used into the desktop application.
    """


    def __init__(self, connectivity, default_model=None,
                 default_integrator=None, compute_phase_plane_params=True):
        BaseSpatialContext.__init__(self, connectivity, default_model, default_integrator)
        self.prepared_model_parameter_names = self._prepare_parameter_names(self.model_parameter_names)

        model = self._get_model_for_region(0)
        if compute_phase_plane_params:
            self._phase_plane = PhasePlaneInteractive(deepcopy(model), deepcopy(self.default_integrator))
            self.phase_plane_params = self._phase_plane.draw_phase_plane()


    @property
    def model_name(self):
        """
        Get class-name for the simulation model saved in current context.
        """
        return self.default_model.__class__.__name__


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


    def load_model_for_connectivity_node(self, connectivity_node_index):
        """
        Sets the parameters of the model found into the phase_plane instance
        to the parameters of the model of the specified connectivity node.
        """
        connectivity_node_index = int(connectivity_node_index)
        model = self._get_model_for_region(connectivity_node_index)
        self._phase_plane.update_all_model_parameters(model)


    def update_model_parameter(self, connectivity_node_index, param_name, param_value):
        """
        Updates the given parameter of the model used by the given connectivity node.
        """
        connectivity_node_index = int(connectivity_node_index)
        if connectivity_node_index < 0:
            return

        param_value = float(param_value)
        model = self._get_model_for_region(connectivity_node_index)
        setattr(model, param_name, numpy.array([param_value]))
        self._phase_plane.update_model_parameter(param_name, param_value)


    def set_model_for_connectivity_nodes(self, source_node_index_for_model, connectivity_node_indexes):
        """
        Set the model for all the nodes specified in 'connectivity_node_indexes' list to
        the model of the node with index 'source_node_index_for_model'
        """
        source_node_index = int(source_node_index_for_model)
        model = self._get_model_for_region(source_node_index)
        for index in connectivity_node_indexes:
            self.connectivity_models[int(index)] = deepcopy(model)


    def reset_model_parameters_for_nodes(self, connectivity_node_indexes):
        """
        Resets all the model parameters for each node specified in the 'connectivity_node_indexes'
        list. The parameters will be reset to their default values.
        """
        if not len(connectivity_node_indexes):
            return
        for index in connectivity_node_indexes:
            index = int(index)
            if index < 0:
                continue
            model = self._compute_default_model(index)
            self.connectivity_models[index] = deepcopy(model)


    def get_data_for_param_sliders(self, connectivity_node_index):
        """
        NOTE: This method may throw 'ValueError' exception.

        Return a dict which contains all the needed information for
        drawing the sliders for the model parameters of the selected connectivity node.
        """
        connectivity_node_index = int(connectivity_node_index)
        current_model = self._get_model_for_region(connectivity_node_index)
        #we have to obtain the model with its default values(the one submitted from burst may have some
        # parameters values changed); the step is computed based on the number of decimals of the default values
        default_model_for_node = parameters_factory.get_traited_instance_for_name(self.model_name,
                                                                                  models_module.Model, {})
        param_sliders_data = dict()
        for param_name in self.model_parameter_names:
            current_value = getattr(current_model, param_name)
            ### Convert to list to avoid having non-serializable values numpy.int32
            if isinstance(current_value, numpy.ndarray):
                current_value = current_value.tolist()[0]
            else:
                #check if the current_value represents a valid number
                #handle the exception in the place where you call this method
                float(current_value)
            ranger = default_model_for_node.trait[param_name].trait.range_interval

            if current_value > ranger.hi:
                current_value = ranger.hi
                self.update_model_parameter(connectivity_node_index, param_name, current_value)
            if current_value < ranger.lo:
                current_value = ranger.lo
                self.update_model_parameter(connectivity_node_index, param_name, current_value)

            param_sliders_data[param_name] = {'min': ranger.lo, 'max': ranger.hi,
                                              'default': current_value, 'step': ranger.step}
        param_sliders_data['all_param_names'] = self.model_parameter_names
        return param_sliders_data


    def get_values_for_parameter(self, parameter_name):
        """
        Returns a list which contains the values for the given model
        parameter name. The values are collected from each node of the used connectivity.

        If all the parameter values are equal then a list with
        one value will be returned.
        """
        default_attr = getattr(self.default_model, parameter_name)
        if isinstance(default_attr, numpy.ndarray):
            default_attr = default_attr.tolist()
        else:
            #if the user set the parameter as a number
            default_attr = [default_attr]

        if len(self.connectivity_models):
            param_values = []
            for i in range(self.connectivity.number_of_regions):
                if i in self.connectivity_models:
                    current_model = self.connectivity_models[i]
                    current_attr = getattr(current_model, parameter_name)
                    if isinstance(current_attr, numpy.ndarray):
                        current_attr = current_attr[0]
                    param_values.append(current_attr)
                else:
                    if 1 < len(default_attr) > i:
                        param_values.append(default_attr[i])
                    else:
                        param_values.append(default_attr[0])
            #check if all the values are equals
            if param_values.count(param_values[0]) != len(param_values):
                return str(param_values)
            else:
                return str([default_attr[0]])

        return deepcopy(str(default_attr))



class SurfaceContextModelParameters(ContextModelParameters):
    """
    This class contains methods which allows you to edit the model
    parameters for each vertex of the given surface.
    """


    def __init__(self, surface, connectivity, default_model=None, default_integrator=None):
        ContextModelParameters.__init__(self, connectivity, default_model, default_integrator, False)
        self.surface = surface
        self.applied_equations = dict()


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
                if vertex_index not in self.applied_equations[model_param][KEY_FOCAL_POINTS]:
                    self.applied_equations[model_param][KEY_FOCAL_POINTS].append(vertex_index)
                    self.applied_equations[model_param][KEY_FOCAL_POINTS_TRIANGLES].append(triangle_index)


    def remove_focal_point(self, model_param, triangle_index):
        """
        NOTE: Expects a vertex index

        Removes a focal point from the list of focal points in which should be
        applied the equation for the given model parameter.
        """
        triangle_index = int(triangle_index)
        if model_param in self.applied_equations:
            if triangle_index >= 0:
                if triangle_index in self.applied_equations[model_param][KEY_FOCAL_POINTS_TRIANGLES]:
                    f_p_idx = self.applied_equations[model_param][KEY_FOCAL_POINTS_TRIANGLES].index(triangle_index)
                    self.applied_equations[model_param][KEY_FOCAL_POINTS].remove(
                        self.applied_equations[model_param][KEY_FOCAL_POINTS][f_p_idx])
                    self.applied_equations[model_param][KEY_FOCAL_POINTS_TRIANGLES].remove(triangle_index)


    def reset_equations_for_all_parameters(self):
        """
        Reset the equations for all the model parameters.
        """
        self.applied_equations = dict()


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
        if parameter_name in self.applied_equations and KEY_EQUATION in self.applied_equations[parameter_name]:
            return self.applied_equations[parameter_name][KEY_EQUATION]
        return None


    def get_focal_points_for_parameter(self, parameter_name):
        """
        :returns: the list of focal points for the equation applied in the given model param.
        """
        if parameter_name in self.applied_equations and KEY_FOCAL_POINTS in self.applied_equations[parameter_name]:
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
    
    