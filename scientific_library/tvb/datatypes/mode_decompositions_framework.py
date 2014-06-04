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
Framework methods for the Mode Decomposition datatypes.

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Paula Sanz Leon <Paula@tvb.invalid>

"""
import numpy
import tvb.datatypes.mode_decompositions_data as mode_decompositions_data


class PrincipalComponentsFramework(mode_decompositions_data.PrincipalComponentsData):
    """
    This class exists to add framework methods to PrincipalComponentsData.
    """
    __tablename__ = None
    
    
    def write_data_slice(self, partial_result):
        """
        Append chunk.
        """
        self.store_data_chunk('weights', partial_result.weights, grow_dimension=2, close_file=False)
        
        self.store_data_chunk('fractions', partial_result.fractions, grow_dimension=1, close_file=False)
        
        partial_result.compute_norm_source()
        self.store_data_chunk('norm_source', partial_result.norm_source, grow_dimension=1, close_file=False)
        
        partial_result.compute_component_time_series()
        self.store_data_chunk('component_time_series', partial_result.component_time_series,
                              grow_dimension=1, close_file=False)
        
        partial_result.compute_normalised_component_time_series()
        self.store_data_chunk('normalised_component_time_series', partial_result.normalised_component_time_series,
                              grow_dimension=1, close_file=False)


    def read_fractions_data(self, from_comp, to_comp):
        """
        Return a list with fractions for components in interval from_comp, to_comp and in
        addition have in position n the sum of the fractions for the rest of the components.
        """
        from_comp = int(from_comp)
        to_comp = int(to_comp)
        all_data = self.get_data('fractions').flat
        sum_others = 0
        for idx, val in enumerate(all_data):
            if idx < from_comp or idx > to_comp:
                sum_others += val
        return numpy.array(all_data[from_comp:to_comp].tolist() + [sum_others])


    def read_weights_data(self, from_comp, to_comp):
        """
        Return the weights data for the components in the interval [from_comp, to_comp].
        """
        from_comp = int(from_comp)
        to_comp = int(to_comp)
        data_slice = slice(from_comp, to_comp, None)
        weights_shape = self.get_data_shape('weights')
        weights_slice = [slice(size) for size in weights_shape]
        weights_slice[0] = data_slice
        weights_data = self.get_data('weights', tuple(weights_slice))
        return weights_data.flatten()


class IndependentComponentsFramework(mode_decompositions_data.IndependentComponentsData):
    """
    This class exists to add framework methods to IndependentComponentsData.
    """
    __tablename__ = None
    
    def write_data_slice(self, partial_result):
        """
        Append chunk.
        """
        
        self.store_data_chunk('unmixing_matrix', partial_result.unmixing_matrix, grow_dimension=2, close_file=False)
                              
        self.store_data_chunk('prewhitening_matrix', partial_result.prewhitening_matrix,
                              grow_dimension=2, close_file=False)

        partial_result.compute_norm_source()
        self.store_data_chunk('norm_source', partial_result.norm_source, grow_dimension=1, close_file=False)
        
        partial_result.compute_component_time_series()
        self.store_data_chunk('component_time_series', partial_result.component_time_series,
                              grow_dimension=1, close_file=False)
        
        partial_result.compute_normalised_component_time_series()
        self.store_data_chunk('normalised_component_time_series', partial_result.normalised_component_time_series,
                              grow_dimension=1, close_file=False)
        
        partial_result.compute_mixing_matrix()
        self.store_data_chunk('mixing_matrix', partial_result.mixing_matrix, grow_dimension=2, close_file=False)


