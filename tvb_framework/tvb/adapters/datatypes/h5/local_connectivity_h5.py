# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need to download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2023, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
# When using The Virtual Brain for scientific publications, please cite it as explained here:
# https://www.thevirtualbrain.org/tvb/zwei/neuroscience-publications
#
#
from tvb.core.neotraits.h5 import H5File, Scalar, Reference, SparseMatrix, EquationScalar
from tvb.datatypes.local_connectivity import LocalConnectivity


class LocalConnectivityH5(H5File):
    def __init__(self, path):
        super(LocalConnectivityH5, self).__init__(path)
        self.surface = Reference(LocalConnectivity.surface, self)
        self.matrix = SparseMatrix(LocalConnectivity.matrix, self)
        self.equation = EquationScalar(LocalConnectivity.equation, self)
        self.cutoff = Scalar(LocalConnectivity.cutoff, self)

    def store(self, datatype, scalars_only=False, store_references=True):
        # type: (LocalConnectivity, bool, bool) -> None
        super(LocalConnectivityH5, self).store(datatype, scalars_only, store_references)

    def load_into(self, datatype):
        # type: (LocalConnectivity) -> None
        super(LocalConnectivityH5, self).load_into(datatype)

    def get_min_max_values(self):
        metadata = self.matrix.get_metadata()
        return metadata.min, metadata.max
