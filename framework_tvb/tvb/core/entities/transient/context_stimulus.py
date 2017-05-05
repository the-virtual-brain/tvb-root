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
.. moduleauthor:: bogdan.neacsa <bogdan.neacsa@codemart.ro>
"""

import json
from tvb.core.adapters.abcadapter import ABCAdapter

SURFACE_PARAMETER = 'surface'
CONNECTIVITY_PARAMETER = 'connectivity'
FOCAL_POINTS_PARAMETER = 'focal_points_triangles'
SCALING_PARAMETER = 'weight'


class SurfaceStimulusContext():
    
    def __init__(self):
        self.selected_stimulus = None
        self.equation_kwargs = {}
        self.focal_points_list = []
    
    def get_session_surface(self):
        return self.equation_kwargs.get(SURFACE_PARAMETER)
    
    def get_selected_stimulus(self):
        return self.selected_stimulus
    
    def set_active_stimulus(self, stimulus_gid):
        self.selected_stimulus = stimulus_gid
    
    def set_focal_points(self, focal_points_json):
        self.focal_points_list = json.loads(focal_points_json)
        self.equation_kwargs['focal_points_surface'] = focal_points_json
        self.equation_kwargs['focal_points_triangles'] = focal_points_json
        
    def update_eq_kwargs(self, new_eq_kwargs):
        """
        We need this since when you pass from step1 -> step2 you don't want to remove
        the focal_points previously defined, so we can't just reassign equation_kwargs = new_eq_kwargs
        but we can't use update either since that would leave the collapsed dictionaries from a previosly
        loaded stimulus.
        """
        previous_keys = self.equation_kwargs.keys()
        for entry in previous_keys:
            if entry not in new_eq_kwargs.keys() and entry not in ('focal_points_surface', 'focal_points_triangles'):
                del self.equation_kwargs[entry]
        for entry in new_eq_kwargs:
            self.equation_kwargs[entry] = new_eq_kwargs[entry]
    
    def update_from_interface(self, stimuli_interface):
        """
        From a stimulus interface, update this context.
        """
        for entry in stimuli_interface:
            #Update this entry
            if (ABCAdapter.KEY_DEFAULT in entry and (entry[ABCAdapter.KEY_TYPE] != 'dict' or 
                                                     (len(entry[ABCAdapter.KEY_ATTRIBUTES]) == 0
                                                      and entry[ABCAdapter.KEY_TYPE] == 'dict'))):
                self.equation_kwargs[entry[ABCAdapter.KEY_NAME]] = entry[ABCAdapter.KEY_DEFAULT]
            #To cover more complex trees
            if ABCAdapter.KEY_OPTIONS in entry:
                for option in entry[ABCAdapter.KEY_OPTIONS]:
                    if (ABCAdapter.KEY_ATTRIBUTES in option and
                            (ABCAdapter.KEY_DEFAULT not in entry or
                             entry[ABCAdapter.KEY_DEFAULT] == option[ABCAdapter.KEY_VALUE])):
                        self.update_from_interface(option[ABCAdapter.KEY_ATTRIBUTES])
            #To cover equation parameters that are of type dict
            if ABCAdapter.KEY_ATTRIBUTES in entry and entry[ABCAdapter.KEY_ATTRIBUTES]:
                self.update_from_interface(entry[ABCAdapter.KEY_ATTRIBUTES])
            if FOCAL_POINTS_PARAMETER == entry[ABCAdapter.KEY_NAME]:
                self.focal_points_list = eval(entry[ABCAdapter.KEY_DEFAULT])
                
    def reset(self):
        self.equation_kwargs = {}
        self.focal_points_list = []
        self.selected_stimulus = None
        

class RegionStimulusContext():
    
    
    def __init__(self):
        self.selected_stimulus = None
        self.equation_kwargs = {}
        self.selected_regions = []
    
    def reset(self):
        self.selected_stimulus = None
        self.equation_kwargs = {}
        self.selected_regions = []
        
    def get_session_connectivity(self):
        return self.equation_kwargs.get(CONNECTIVITY_PARAMETER)
 
    def get_weights(self):
        return self.selected_regions
    
    def set_weights(self, new_weights):
        self.selected_regions = new_weights
    
    def set_active_stimulus(self, stimulus_gid):
        self.selected_stimulus = stimulus_gid
        
    def update_from_interface(self, stimuli_interface):
        """
        From a stimulus interface, update this context.
        """
        for entry in stimuli_interface:
            #Update this entry
            if (ABCAdapter.KEY_DEFAULT in entry and (entry[ABCAdapter.KEY_TYPE] != 'dict' or 
                                                     (len(entry[ABCAdapter.KEY_ATTRIBUTES]) == 0
                                                      and entry[ABCAdapter.KEY_TYPE] == 'dict'))):
                self.equation_kwargs[entry[ABCAdapter.KEY_NAME]] = entry[ABCAdapter.KEY_DEFAULT]
            #To cover more complex trees
            if ABCAdapter.KEY_OPTIONS in entry:
                for option in entry[ABCAdapter.KEY_OPTIONS]:
                    if (ABCAdapter.KEY_ATTRIBUTES in option and 
                        (ABCAdapter.KEY_DEFAULT not in entry
                         or entry[ABCAdapter.KEY_DEFAULT] == option[ABCAdapter.KEY_VALUE])):
                        self.update_from_interface(option[ABCAdapter.KEY_ATTRIBUTES])
            #To cover equation parameters that are of type dict
            if ABCAdapter.KEY_ATTRIBUTES in entry and entry[ABCAdapter.KEY_ATTRIBUTES]:
                self.update_from_interface(entry[ABCAdapter.KEY_ATTRIBUTES])
            if SCALING_PARAMETER == entry[ABCAdapter.KEY_NAME]:
                self.selected_regions = eval(entry[ABCAdapter.KEY_DEFAULT])
