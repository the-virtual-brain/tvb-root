# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2020, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
from tvb.core.neotraits.h5 import H5File, DataSet, Reference, Json
from tvb.datatypes.temporal_correlations import CrossCorrelation


class CrossCorrelationH5(H5File):

    def __init__(self, path):
        super(CrossCorrelationH5, self).__init__(path)
        self.array_data = DataSet(CrossCorrelation.array_data, self, expand_dimension=3)
        self.source = Reference(CrossCorrelation.source, self)
        self.time = DataSet(CrossCorrelation.time, self)
        self.labels_ordering = Json(CrossCorrelation.labels_ordering, self)

    def read_data_shape(self):
        """
        The shape of the data
        """
        return self.array_data.shape

    def read_data_slice(self, data_slice):
        """
        Expose chunked-data access.
        """
        return self.array_data[data_slice]

    def write_data_slice(self, partial_result):
        """
        Append chunk.
        """
        self.array_data.append(partial_result.array_data)
