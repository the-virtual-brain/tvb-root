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

The Data component of Graph datatypes.

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>
.. moduleauthor:: Paula Sanz Leon <paula.sanz-leon@univ-amu.fr>

"""

import tvb.basic.traits.core as core
import tvb.basic.traits.types_basic as basic
import tvb.datatypes.arrays as arrays
import tvb.datatypes.time_series as time_series
import tvb.datatypes.connectivity as connectivity



class CovarianceData(arrays.MappedArray):
    """
    Result of a Covariance  Analysis.
    """

    array_data = arrays.ComplexArray(file_storage=core.FILE_STORAGE_EXPAND)

    source = time_series.TimeSeries(
        label="Source time-series",
        doc="Links to the time-series on which NodeCovariance is applied.")

    __generate_table__ = True



class CorrelationCoefficientsData(arrays.MappedArray):

    array_data = arrays.FloatArray(file_storage=core.FILE_STORAGE_DEFAULT)

    source = time_series.TimeSeries(
        label="Source time-series",
        doc="Links to the time-series on which Correlation (coefficients) is applied.")

    labels_ordering = basic.List(
        label="Dimension Names",
        default=["Node", "Node", "State Variable", "Mode"],
        doc="""List of strings representing names of each data dimension""")

    __generate_table__ = True



class ConnectivityMeasureData(arrays.MappedArray):
    """
    An array representing a measure of a Connectivity dataType.
    """

    connectivity = connectivity.Connectivity

    __generate_table__ = True
    
    