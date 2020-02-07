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
from tvb.basic.neotraits.api import Attr
from tvb.datatypes.graph import Covariance, CorrelationCoefficients, ConnectivityMeasure
from tvb.adapters.datatypes.h5.spectral_h5 import DataTypeMatrixH5
from tvb.core.neotraits.h5 import DataSet, Reference, Json, Scalar


class CovarianceH5(DataTypeMatrixH5):

    def __init__(self, path):
        super(CovarianceH5, self).__init__(path)
        self.array_data = DataSet(Covariance.array_data, self, expand_dimension=2)
        self.source = Reference(Covariance.source, self)

    def write_data_slice(self, partial_result):
        """
        Append chunk.
        """
        self.array_data.append(partial_result, close_file=False)


class CorrelationCoefficientsH5(DataTypeMatrixH5):

    def __init__(self, path):
        super(CorrelationCoefficientsH5, self).__init__(path)
        self.array_data = DataSet(CorrelationCoefficients.array_data, self)
        self.source = Reference(CorrelationCoefficients.source, self)
        self.labels_ordering = Json(CorrelationCoefficients.labels_ordering, self)

    def get_correlation_data(self, selected_state, selected_mode):
        matrix_to_display = self.array_data[:, :, int(selected_state) - 1, int(selected_mode)]
        return list(matrix_to_display.flat)


class ConnectivityMeasureH5(DataTypeMatrixH5):

    def __init__(self, path):
        super(ConnectivityMeasureH5, self).__init__(path)
        self.array_data = DataSet(ConnectivityMeasure.array_data, self)
        self.connectivity = Reference(ConnectivityMeasure.connectivity, self)
        self.title = Scalar(Attr(str), self, name='title')

    def get_array_data(self):
        return self.array_data[:]
