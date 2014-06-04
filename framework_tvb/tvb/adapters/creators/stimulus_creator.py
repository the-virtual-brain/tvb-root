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
.. Ionel Ortelecan <ionel.ortelecan@codemart.ro>
"""

from tvb.core.adapters.abcadapter import ABCSynchronous
from tvb.datatypes.patterns import StimuliSurface, StimuliRegion
from tvb.datatypes.equations import Equation
import tvb.basic.traits.traited_interface as interface



class SurfaceStimulusCreator(ABCSynchronous):
    """
    The purpose of this adapter is to create a StimuliSurface.
    """


    def get_input_tree(self):
        """
        Returns the input interface for this adapter.
        """
        stimuli_surface = StimuliSurface()
        stimuli_surface.trait.bound = interface.INTERFACE_ATTRIBUTES_ONLY
        inputList = stimuli_surface.interface[interface.INTERFACE_ATTRIBUTES]

        return inputList


    def get_output(self):
        """
        Describes the outputs of the launch method.
        """
        return [StimuliSurface]


    def launch(self, **kwargs):
        """
        Used for creating a `StimuliSurface` instance
        """
        stimuli_surface = StimuliSurface(storage_path=self.storage_path)
        stimuli_surface.surface = kwargs['surface']
        triangles_indices = kwargs['focal_points_triangles']
        focal_points = []
        fp_triangle_indices = []
        for triangle_index in triangles_indices:
            focal_points.append(int(stimuli_surface.surface.triangles[triangle_index][0]))
            fp_triangle_indices.append(int(triangle_index))
        stimuli_surface.focal_points_triangles = fp_triangle_indices
        stimuli_surface.focal_points_surface = focal_points
        stimuli_surface.spatial = self.get_spatial_equation(kwargs)
        stimuli_surface.temporal = self.get_temporal_equation(kwargs)

        return stimuli_surface


    def get_spatial_equation(self, kwargs):
        """
        From a dictionary of arguments build the spatial equation.
        """
        return Equation.build_equation_from_dict('spatial', kwargs)


    def get_temporal_equation(self, kwargs):
        """
        From a dictionary of arguments build the temporal equation.
        """
        return Equation.build_equation_from_dict('temporal', kwargs)


    def get_required_memory_size(self, **kwargs):
        """
        Return the required memory to run this algorithm.
        """
        return -1


    def get_required_disk_size(self, **kwargs):
        """
        Returns the required disk size to be able to run the adapter. (in kB)
        """
        return 0



class RegionStimulusCreator(ABCSynchronous):
    """
    The purpose of this adapter is to create a StimuliRegion.
    """


    def get_input_tree(self):
        """
        Returns the input interface for this adapter.
        """
        stimuli_region = StimuliRegion()
        stimuli_region.trait.bound = interface.INTERFACE_ATTRIBUTES_ONLY
        inputList = stimuli_region.interface[interface.INTERFACE_ATTRIBUTES]

        return inputList


    def get_output(self):
        """
        Describes the outputs of the launch method.
        """
        return [StimuliRegion]


    def launch(self, **kwargs):
        """
        Used for creating a `StimuliRegion` instance
        """
        stimuli_region = StimuliRegion(storage_path=self.storage_path)
        stimuli_region.connectivity = kwargs['connectivity']
        stimuli_region.weight = kwargs['weight']
        stimuli_region.temporal = Equation.build_equation_from_dict('temporal', kwargs)

        return stimuli_region


    def get_required_disk_size(self, **kwargs):
        """
        Returns the required disk size to be able to run the adapter. (in kB)
        """
        return 0


    def get_required_memory_size(self, **kwargs):
        """
        Return the required memory to run this algorithm.
        """
        return -1
    
    
    