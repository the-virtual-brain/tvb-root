# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and 
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
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

The Graph datatypes. This brings together the scientific and framework methods
that are associated with the Graph datatypes.

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>
.. moduleauthor:: Paula Sanz Leon <paula.sanz-leon@univ-amu.fr>

"""

from tvb.basic.traits import core, types_basic as basic
from tvb.basic.logger.builder import get_logger
from tvb.datatypes import arrays, time_series, connectivity

LOG = get_logger(__name__)


class Covariance(arrays.MappedArray):
    """Covariance datatype."""

    array_data = arrays.ComplexArray(file_storage=core.FILE_STORAGE_EXPAND)

    source = time_series.TimeSeries(
        label="Source time-series",
        doc="Links to the time-series on which NodeCovariance is applied.")

    __generate_table__ = True

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

    def _find_summary_info(self):
        summary = {"Graph type": self.__class__.__name__,
                   "Source": self.source.title}

        summary.update(self.get_info_about_array('array_data'))
        return summary


class CorrelationCoefficients(arrays.MappedArray):
    """Correlation coefficients datatype."""

    # Extreme values for pearson Correlation Coefficients
    PEARSON_MIN = -1
    PEARSON_MAX = 1

    array_data = arrays.FloatArray(file_storage=core.FILE_STORAGE_DEFAULT)

    source = time_series.TimeSeries(
        label="Source time-series",
        doc="Links to the time-series on which Correlation (coefficients) is applied.")

    labels_ordering = basic.List(
        label="Dimension Names",
        default=["Node", "Node", "State Variable", "Mode"],
        doc="""List of strings representing names of each data dimension""")

    __generate_table__ = True

    def configure(self):
        """After populating few fields, compute the rest of the fields"""
        # Do not call super, because that accesses data not-chunked
        self.nr_dimensions = len(self.read_data_shape())
        for i in range(self.nr_dimensions):
            setattr(self, 'length_%dd' % (i + 1), int(self.read_data_shape()[i]))

    def _find_summary_info(self):
        summary = {"Graph type": self.__class__.__name__,
                   "Source": self.source.title,
                   "Dimensions": self.labels_ordering}
        summary.update(self.get_info_about_array('array_data'))
        return summary

    def get_correlation_data(self, selected_state, selected_mode):
        matrix_to_display = self.array_data[:, :, int(selected_state), int(selected_mode)]
        return list(matrix_to_display.flat)


class ConnectivityMeasure(arrays.MappedArray):
    """Measurement of based on a connectivity."""

    connectivity = connectivity.Connectivity

    def _find_summary_info(self):
        summary = {"Graph type": self.__class__.__name__}
        # summary["Source"] = self.connectivity.title
        summary.update(self.get_info_about_array('array_data'))
        return summary

    __generate_table__ = True
