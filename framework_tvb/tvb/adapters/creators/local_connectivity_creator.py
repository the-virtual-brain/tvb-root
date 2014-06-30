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

from tvb.basic.config.settings import TVBSettings
from tvb.core.adapters.abcadapter import ABCAsynchronous
from tvb.datatypes.surfaces import LocalConnectivity
from tvb.datatypes.equations import Equation
import tvb.basic.traits.traited_interface as interface



class LocalConnectivityCreator(ABCAsynchronous):
    """
    The purpose of this adapter is create a LocalConnectivity.
    """

    def get_input_tree(self):
        """
        Returns the input interface for this adapter.
        """
        local_connectivity = LocalConnectivity()
        local_connectivity.trait.bound = interface.INTERFACE_ATTRIBUTES_ONLY
        inputList = local_connectivity.interface[interface.INTERFACE_ATTRIBUTES]

        return inputList


    def get_output(self):
        """
        Describes the outputs of the launch method.
        """
        return [LocalConnectivity]


    def launch(self, **kwargs):
        """
        Used for creating a `LocalConnectivity`
        """
        local_connectivity = LocalConnectivity(storage_path=self.storage_path)
        local_connectivity.cutoff = float(kwargs['cutoff'])
        local_connectivity.surface = kwargs['surface']
        local_connectivity.equation = self.get_lconn_equation(kwargs)
        local_connectivity.compute_sparse_matrix()

        return local_connectivity

    
    def get_lconn_equation(self, kwargs):
        """
        Get the equation for the local connectivity from a dictionary of arguments.
        """
        return Equation.build_equation_from_dict('equation', kwargs)


    def get_required_disk_size(self, **kwargs):
        """
        Returns the required disk size to be able to run the adapter. (in kB)
        """
        if 'surface' in kwargs:
            surface = kwargs['surface']
            points_no = float(kwargs['cutoff']) / surface.edge_length_mean
            disk_size_b = surface.number_of_vertices * points_no * points_no * 8
            return disk_size_b * TVBSettings.MAGIC_NUMBER / 2 ** 10
        return 0


    def get_required_memory_size(self, **kwargs):
        """
        Return the required memory to run this algorithm.
        """
        if 'surface' in kwargs:
            surface = kwargs['surface']
            return surface.number_of_vertices * surface.number_of_vertices * 8.0
        return -1



    
    
    