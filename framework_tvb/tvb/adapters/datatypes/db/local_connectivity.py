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
import scipy.sparse
from sqlalchemy import Column, Integer, ForeignKey, Float, String
from sqlalchemy.orm import relationship
from tvb.datatypes.local_connectivity import LocalConnectivity
from tvb.adapters.datatypes.db.surface import SurfaceIndex
from tvb.core.entities.model.model_datatype import DataType
from tvb.core.neotraits.db import from_ndarray


class LocalConnectivityIndex(DataType):
    id = Column(Integer, ForeignKey(DataType.id), primary_key=True)

    surface_gid = Column(String(32), ForeignKey(SurfaceIndex.gid), nullable=not LocalConnectivity.surface.required)
    surface = relationship(SurfaceIndex, foreign_keys=surface_gid, primaryjoin=SurfaceIndex.gid == surface_gid)

    matrix_non_zero_min = Column(Float)
    matrix_non_zero_max = Column(Float)
    matrix_non_zero_mean = Column(Float)

    def fill_from_has_traits(self, datatype):
        # type: (LocalConnectivity)  -> None
        super(LocalConnectivityIndex, self).fill_from_has_traits(datatype)
        I, J, V = scipy.sparse.find(datatype.matrix)
        self.matrix_non_zero_min, self.matrix_non_zero_max, self.matrix_non_zero_mean = from_ndarray(V)
        self.surface_gid = datatype.surface.gid.hex
