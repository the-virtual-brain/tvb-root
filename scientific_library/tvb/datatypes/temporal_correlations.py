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

The Temporal Correlation datatypes. This brings together the scientific and
framework methods that are associated with the Temporal Correlation datatypes.

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>

"""

import tvb.basic.traits.core as core
import tvb.basic.traits.types_basic as basic
import tvb.datatypes.arrays as arrays
import tvb.datatypes.time_series as time_series
from tvb.basic.logger.builder import get_logger
from tvb.basic.traits.types_mapped import MappedType

LOG = get_logger(__name__)


class CrossCorrelation(MappedType):
    """
    Result of a CrossCorrelation Analysis.
    """
    array_data = arrays.FloatArray(file_storage=core.FILE_STORAGE_EXPAND)

    source = time_series.TimeSeries(
        label="Source time-series",
        doc="""Links to the time-series on which the cross_correlation is applied.""")

    time = arrays.FloatArray(label="Temporal Offsets")

    labels_ordering = basic.List(
        label="Dimension Names",
        default=["Offsets", "Node", "Node", "State Variable", "Mode"],
        doc="""List of strings representing names of each data dimension""")

    def configure(self):
        """After populating few fields, compute the rest of the fields"""
        # Do not call super, because that accesses data not-chunked
        self.nr_dimensions = len(self.read_data_shape())
        for i in range(self.nr_dimensions):
            setattr(self, 'length_%dd' % (i + 1), int(self.read_data_shape()[i]))

    def read_data_shape(self):
        """
        Expose shape read on field 'data'
        """
        return self.get_data_shape('array_data')

    def read_data_slice(self, data_slice):
        """
        Expose chunked-data access.
        """
        return self.get_data('array_data', data_slice)

    def write_data_slice(self, partial_result):
        """
        Append chunk.
        """
        self.store_data_chunk('array_data', partial_result.array_data, grow_dimension=3, close_file=False)

    def _find_summary_info(self):
        """
        Gather scientifically interesting summary information from an instance of this datatype.
        """
        summary = {"Temporal correlation type": self.__class__.__name__,
                   "Source": self.source.title,
                   "Dimensions": self.labels_ordering}

        summary.update(self.get_info_about_array('array_data'))
        return summary
