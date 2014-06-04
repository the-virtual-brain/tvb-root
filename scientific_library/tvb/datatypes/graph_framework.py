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

Framework methods for the Graph datatypes.

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Paula Sanz Leon <paula.sanz-leon@univ-amu.fr>
"""

import tvb.datatypes.graph_data as graph_data


class CovarianceFramework(graph_data.CovarianceData):
    """
    This class exists to add framework methods to CovarianceData.
    """
    __tablename__ = None


    def configure(self):
        """After populating few fields, compute the rest of the fields"""
        # Do not call super, because that accesses data not-chunked
        self.nr_dimensions = len(self.read_data_shape())
        for i in range(self.nr_dimensions):
            setattr(self, 'length_%dd' % (i + 1), int(self.read_data_shape()[i]))


    def write_data_slice(self, partial_result):
        """
        Append chunk.
        """
        self.store_data_chunk('array_data', partial_result, grow_dimension=2, close_file=False)



class CorrelationCoefficientsFramework(graph_data.CorrelationCoefficientsData):
    """
    This class exists to add framework methods to CovarianceData.
    """
    __tablename__ = None


    def configure(self):
        """After populating few fields, compute the rest of the fields"""
        # Do not call super, because that accesses data not-chunked
        self.nr_dimensions = len(self.read_data_shape())
        for i in range(self.nr_dimensions):
            setattr(self, 'length_%dd' % (i + 1), int(self.read_data_shape()[i]))



class ConnectivityMeasureFramework(graph_data.ConnectivityMeasureData):
    """
    Framework methods for ConnectivityMeasure entity
    """
    __tablename__ = None

